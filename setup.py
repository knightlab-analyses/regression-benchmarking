# ----------------------------------------------------------------------------
# Copyright (c) 2020, mlab development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools import setup, find_packages
import versioneer

setup(
    name="q2-mlab",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    author="Imran McGrath",
    author_email="pmcgrath@ucsd.edu",
    description="Machine learning benchmarking in Qiime",
    license='BSD-3-Clause',
    url="https://qiime2.org",
    entry_points={
        'qiime2.plugins': ['q2-mlab=q2_mlab.plugin_setup:plugin'],
        'console_scripts': ['orchestrator=q2_mlab.orchestrator:cli'],
    },
    package_data={'q2_mlab': ['assets/index.html', 
                              'citations.bib',
                              'templates/array_job_template.sh']},
    zip_safe=False,
    install_requires=['xgboost', 'calour', 'click']
)
