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



def project_from_config_norender(cfg, packages=None, path='/invalid-root-path', verify_version=False):
    if packages is None:
        packages = {}
    partial = dbt.config.project.PartialProject.from_dicts(
        path,
        project_dict=cfg,
        packages_dict=packages,
        selectors_dict={},
        verify_version=verify_version,
    )
    # no rendering
    rendered = dbt.config.project.RenderComponents(
        project_dict=partial.project_dict,
        packages_dict=partial.packages_dict,
        selectors_dict=partial.selectors_dict,
    )
    return partial.create_project(rendered)


def project_from_config_rendered(cfg, packages=None, path='/invalid-root-path', verify_version=False):
    if packages is None:
        packages = {}
    partial = dbt.config.project.PartialProject.from_dicts(
        path,
        project_dict=cfg,
        packages_dict=packages,
        selectors_dict={},
        verify_version=verify_version,
    )
    return partial.render(empty_project_renderer())



class TestRuntimeConfig(BaseConfigTest):
    def setUp(self):
        self.profiles_dir = '/invalid-profiles-path'
        self.project_dir = '/invalid-root-path'
        super().setUp()
        self.default_project_data['project-root'] = self.project_dir

    def get_project(self):
        return project_from_config_norender(self.default_project_data, verify_version=self.args.version_check)

    def get_profile(self):
        renderer = empty_profile_renderer()
        return dbt.config.Profile.from_raw_profiles(
            self.default_profile_data, self.default_project_data['profile'], renderer
        )

    def from_parts(self, exc=None):
        with self.assertRaisesOrReturns(exc) as err:
            project = self.get_project()
            profile = self.get_profile()

            result = dbt.config.RuntimeConfig.from_parts(project, profile, self.args)

        if exc is None:
            return result
        else:
            return err

    def test_from_parts(self):
        project = self.get_project()
        profile = self.get_profile()
        config = dbt.config.RuntimeConfig.from_parts(project, profile, self.args)
        print(f"----- in test_from_parts: {config}")

        self.assertEqual(config.cli_vars, {})
        self.assertEqual(config.to_profile_info(), profile.to_profile_info())
        # we should have the default quoting set in the full config, but not in
        # the project
        # TODO(jeb): Adapters must assert that quoting is populated?
        expected_project = project.to_project_config()
        self.assertEqual(expected_project['quoting'], {})

        expected_project['quoting'] = {
            'database': True,
            'identifier': True,
            'schema': True,
        }
        self.assertEqual(config.to_project_config(), expected_project)

    def test_str(self):
        project = self.get_project()
        profile = self.get_profile()
        config = dbt.config.RuntimeConfig.from_parts(project, profile, {})

        # to make sure nothing terrible happens
        str(config)

    def test_validate_fails(self):
        project = self.get_project()
        profile = self.get_profile()
        # invalid - must be boolean
        profile.config.use_colors = 100
        with self.assertRaises(dbt.exceptions.DbtProjectError):
            dbt.config.RuntimeConfig.from_parts(project, profile, {})

    def test_supported_version(self):
        self.default_project_data['require-dbt-version'] = '>0.0.0'
        conf = self.from_parts()
        self.assertEqual(set(x.to_version_string() for x in conf.dbt_version), {'>0.0.0'})

    def test_unsupported_version(self):
        self.default_project_data['require-dbt-version'] = '>99999.0.0'
        raised = self.from_parts(dbt.exceptions.DbtProjectError)
        self.assertIn('This version of dbt is not supported', str(raised.exception))

    def test_unsupported_version_no_check(self):
        self.default_project_data['require-dbt-version'] = '>99999.0.0'
        self.args.version_check = False
        conf = self.from_parts()
        self.assertEqual(set(x.to_version_string() for x in conf.dbt_version), {'>99999.0.0'})

    def test_supported_version_range(self):
        self.default_project_data['require-dbt-version'] = ['>0.0.0', '<=99999.0.0']
        conf = self.from_parts()
        self.assertEqual(set(x.to_version_string() for x in conf.dbt_version), {'>0.0.0', '<=99999.0.0'})

    def test_unsupported_version_range(self):
        self.default_project_data['require-dbt-version'] = ['>0.0.0', '<=0.0.1']
        raised = self.from_parts(dbt.exceptions.DbtProjectError)
        self.assertIn('This version of dbt is not supported', str(raised.exception))

    def test_unsupported_version_range_bad_config(self):
        self.default_project_data['require-dbt-version'] = ['>0.0.0', '<=0.0.1']
        self.default_project_data['some-extra-field-not-allowed'] = True
        raised = self.from_parts(dbt.exceptions.DbtProjectError)
        self.assertIn('This version of dbt is not supported', str(raised.exception))

    def test_unsupported_version_range_no_check(self):
        self.default_project_data['require-dbt-version'] = ['>0.0.0', '<=0.0.1']
        self.args.version_check = False
        conf = self.from_parts()
        self.assertEqual(set(x.to_version_string() for x in conf.dbt_version), {'>0.0.0', '<=0.0.1'})

    def test_impossible_version_range(self):
        self.default_project_data['require-dbt-version'] = ['>99999.0.0', '<=0.0.1']
        raised = self.from_parts(dbt.exceptions.DbtProjectError)
        self.assertIn('The package version requirement can never be satisfied', str(raised.exception))

    def test_unsupported_version_extra_config(self):
        self.default_project_data['some-extra-field-not-allowed'] = True
        raised = self.from_parts(dbt.exceptions.DbtProjectError)
        self.assertIn('Additional properties are not allowed', str(raised.exception))

    def test_archive_not_allowed(self):
        self.default_project_data['archive'] = [{
            "source_schema": 'a',
            "target_schema": 'b',
            "tables": [
                {
                    "source_table": "seed",
                    "target_table": "archive_actual",
                    "updated_at": 'updated_at',
                    "unique_key": '''id || '-' || first_name'''
                },
            ],
        }]
        with self.assertRaises(dbt.exceptions.DbtProjectError):
            self.get_project()

    def test__no_unused_resource_config_paths(self):
        self.default_project_data.update({
            'models': model_config,
            'seeds': {},
        })
        project = self.from_parts()

        resource_fqns = {'models': model_fqns}
        unused = project.get_unused_resource_config_paths(resource_fqns, [])
        self.assertEqual(len(unused), 0)

    def test__unused_resource_config_paths(self):
        self.default_project_data.update({
            'models': model_config['my_package_name'],
            'seeds': {},
        })
        project = self.from_parts()

        resource_fqns = {'models': model_fqns}
        unused = project.get_unused_resource_config_paths(resource_fqns, [])
        self.assertEqual(len(unused), 3)

    def test__get_unused_resource_config_paths_empty(self):
        project = self.from_parts()
        unused = project.get_unused_resource_config_paths({'models': frozenset((
            ('my_test_project', 'foo', 'bar'),
            ('my_test_project', 'foo', 'baz'),
        ))}, [])
        self.assertEqual(len(unused), 0)

    def test__warn_for_unused_resource_config_paths_empty(self):
        project = self.from_parts()
        dbt.flags.WARN_ERROR = True
        try:
            project.warn_for_unused_resource_config_paths({'models': frozenset((
                ('my_test_project', 'foo', 'bar'),
                ('my_test_project', 'foo', 'baz'),
            ))}, [])
        finally:
            dbt.flags.WARN_ERROR = False


