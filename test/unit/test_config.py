from copy import deepcopy
from contextlib import contextmanager
import json
import os
import shutil
import tempfile
import unittest

from unittest import mock
import yaml

import dbt.config
import dbt.exceptions
from dbt.adapters.factory import load_plugin
from dbt.adapters.postgres import PostgresCredentials
from dbt.adapters.redshift import RedshiftCredentials
from dbt.context.base import generate_base_context
from dbt.contracts.connection import QueryComment, DEFAULT_QUERY_COMMENT
from dbt.contracts.project import PackageConfig, LocalPackage, GitPackage
from dbt.node_types import NodeType
from dbt.semver import VersionSpecifier
from dbt.task.run_operation import RunOperationTask

from .utils import normalize, config_from_parts_or_dicts

INITIAL_ROOT = os.getcwd()


@contextmanager
def temp_cd(path):
    current_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current_path)

@contextmanager
def raises_nothing():
    yield


def empty_profile_renderer():
    return dbt.config.renderer.ProfileRenderer(generate_base_context({}))


def empty_project_renderer():
    return dbt.config.renderer.DbtProjectYamlRenderer(generate_base_context({}))


model_config = {
    'my_package_name': {
        'enabled': True,
        'adwords': {
            'adwords_ads': {
                'materialized': 'table',
                'enabled': True,
                'schema': 'analytics'
            }
        },
        'snowplow': {
            'snowplow_sessions': {
                'sort': 'timestamp',
                'materialized': 'incremental',
                'dist': 'user_id',
                'unique_key': 'id'
            },
            'base': {
                'snowplow_events': {
                    'sort': ['timestamp', 'userid'],
                    'materialized': 'table',
                    'sort_type': 'interleaved',
                    'dist': 'userid'
                }
            }
        }
    }
}

model_fqns = frozenset((
    ('my_package_name', 'snowplow', 'snowplow_sessions'),
    ('my_package_name', 'snowplow', 'base', 'snowplow_events'),
    ('my_package_name', 'adwords', 'adwords_ads'),
))


class Args:
    def __init__(self, profiles_dir=None, threads=None, profile=None,
                 cli_vars=None, version_check=None, project_dir=None):
        self.profile = profile
        if threads is not None:
            self.threads = threads
        if profiles_dir is not None:
            self.profiles_dir = profiles_dir
        if cli_vars is not None:
            self.vars = cli_vars
        if version_check is not None:
            self.version_check = version_check
        if project_dir is not None:
            self.project_dir = project_dir


class BaseConfigTest(unittest.TestCase):
    """Subclass this, and before calling the superclass setUp, set
    self.profiles_dir and self.project_dir.
    """
    def setUp(self):
        self.default_project_data = {
            'version': '0.0.1',
            'name': 'my_test_project',
            'profile': 'default',
            'config-version': 2,
        }
        self.default_profile_data = {
            'default': {
                'outputs': {
                    'postgres': {
                        'type': 'postgres',
                        'host': 'postgres-db-hostname',
                        'port': 5555,
                        'user': 'db_user',
                        'pass': 'db_pass',
                        'dbname': 'postgres-db-name',
                        'schema': 'postgres-schema',
                        'threads': 7,
                    },
                    'redshift': {
                        'type': 'redshift',
                        'host': 'redshift-db-hostname',
                        'port': 5555,
                        'user': 'db_user',
                        'pass': 'db_pass',
                        'dbname': 'redshift-db-name',
                        'schema': 'redshift-schema',
                    },
                    'with-vars': {
                        'type': "{{ env_var('env_value_type') }}",
                        'host': "{{ env_var('env_value_host') }}",
                        'port': "{{ env_var('env_value_port') | as_number }}",
                        'user': "{{ env_var('env_value_user') }}",
                        'pass': "{{ env_var('env_value_pass') }}",
                        'dbname': "{{ env_var('env_value_dbname') }}",
                        'schema': "{{ env_var('env_value_schema') }}",
                    },
                    'cli-and-env-vars': {
                        'type': "{{ env_var('env_value_type') }}",
                        'host': "{{ var('cli_value_host') }}",
                        'port': "{{ env_var('env_value_port') | as_number }}",
                        'user': "{{ env_var('env_value_user') }}",
                        'pass': "{{ env_var('env_value_pass') }}",
                        'dbname': "{{ env_var('env_value_dbname') }}",
                        'schema': "{{ env_var('env_value_schema') }}",
                    }
                },
                'target': 'postgres',
            },
            'other': {
                'outputs': {
                    'other-postgres': {
                        'type': 'postgres',
                        'host': 'other-postgres-db-hostname',
                        'port': 4444,
                        'user': 'other_db_user',
                        'pass': 'other_db_pass',
                        'dbname': 'other-postgres-db-name',
                        'schema': 'other-postgres-schema',
                        'threads': 2,
                    }
                },
                'target': 'other-postgres',
            },
            'empty_profile_data': {}
        }
        self.args = Args(profiles_dir=self.profiles_dir, cli_vars='{}',
                         version_check=True, project_dir=self.project_dir)
        self.env_override = {
            'env_value_type': 'postgres',
            'env_value_host': 'env-postgres-host',
            'env_value_port': '6543',
            'env_value_user': 'env-postgres-user',
            'env_value_pass': 'env-postgres-pass',
            'env_value_dbname': 'env-postgres-dbname',
            'env_value_schema': 'env-postgres-schema',
            'env_value_profile': 'default',
        }

    def assertRaisesOrReturns(self, exc):
        if exc is None:
            return raises_nothing()
        else:
            return self.assertRaises(exc)


class BaseFileTest(BaseConfigTest):
    def setUp(self):
        self.project_dir = normalize(tempfile.mkdtemp())
        self.profiles_dir = normalize(tempfile.mkdtemp())
        super().setUp()

    def tearDown(self):
        try:
            shutil.rmtree(self.project_dir)
        except EnvironmentError:
            pass
        try:
            shutil.rmtree(self.profiles_dir)
        except EnvironmentError:
            pass

    def project_path(self, name):
        return os.path.join(self.project_dir, name)

    def profile_path(self, name):
        return os.path.join(self.profiles_dir, name)

    def write_project(self, project_data=None):
        if project_data is None:
            project_data = self.project_data
        with open(self.project_path('dbt_project.yml'), 'w') as fp:
            yaml.dump(project_data, fp)

    def write_packages(self, package_data):
        with open(self.project_path('packages.yml'), 'w') as fp:
            yaml.dump(package_data, fp)

    def write_profile(self, profile_data=None):
        if profile_data is None:
            profile_data = self.profile_data
        with open(self.profile_path('profiles.yml'), 'w') as fp:
            yaml.dump(profile_data, fp)

    def write_empty_profile(self):
        with open(self.profile_path('profiles.yml'), 'w') as fp:
            yaml.dump('', fp)


class TestProfile(BaseConfigTest):
    def setUp(self):
        self.profiles_dir = '/invalid-path'
        self.project_dir = '/invalid-project-path'
        super().setUp()

    def from_raw_profiles(self):
        renderer = empty_profile_renderer()
        return dbt.config.Profile.from_raw_profiles(
            self.default_profile_data, 'default', renderer
        )

    def test_from_raw_profiles(self):
        profile = self.from_raw_profiles()
        self.assertEqual(profile.profile_name, 'default')
        self.assertEqual(profile.target_name, 'postgres')
        self.assertEqual(profile.threads, 7)
        self.assertTrue(profile.config.send_anonymous_usage_stats)
        self.assertIsNone(profile.config.use_colors)
        self.assertTrue(isinstance(profile.credentials, PostgresCredentials))
        self.assertEqual(profile.credentials.type, 'postgres')
        self.assertEqual(profile.credentials.host, 'postgres-db-hostname')
        self.assertEqual(profile.credentials.port, 5555)
        self.assertEqual(profile.credentials.user, 'db_user')
        self.assertEqual(profile.credentials.password, 'db_pass')
        self.assertEqual(profile.credentials.schema, 'postgres-schema')
        self.assertEqual(profile.credentials.database, 'postgres-db-name')

