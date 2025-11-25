#!/usr/bin/env python
from setuptools import find_packages, setup
import os
import subprocess
import time

version_file = 'basicsr/version.py'


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_git_hash():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
__gitsha__ = '{}'
version_info = ({})
"""
    sha = get_hash()
    
    # VERSION 파일이 없을 경우 기본값 사용
    if os.path.exists('VERSION'):
        with open('VERSION', 'r') as f:
            SHORT_VERSION = f.read().strip()
    else:
        SHORT_VERSION = '1.0.0'
    
    VERSION_INFO = ', '.join([x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split('.')])
    version_file_str = content.format(time.asctime(), SHORT_VERSION, sha, VERSION_INFO)
    
    with open(version_file, 'w') as f:
        f.write(version_file_str)


def get_version():
    write_version_py()
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = []
        for line in f.readlines():
            line = line.strip()
            # Skip empty lines, comments, and pip options
            if line and not line.startswith('#') and not line.startswith('--'):
                requires.append(line)
    return requires


if __name__ == '__main__':
    # CUDA 확장 컴파일 비활성화 (서버 환경에서는 불필요)
    cuda_ext = os.getenv('BASICSR_EXT', 'False')
    
    if cuda_ext == 'True':
        print("Warning: CUDA extensions are disabled by default for server environments.")
        print("Set BASICSR_EXT=True only if you need CUDA extensions and have proper CUDA setup.")
        ext_modules = []
        setup_kwargs = dict()
    else:
        ext_modules = []
        setup_kwargs = dict()

    setup(
        name='basicsr',
        version=get_version(),
        description='Open Source Image and Video Super-Resolution Toolbox',
        long_description=readme() if os.path.exists('README.md') else 'BasicSR',
        long_description_content_type='text/markdown',
        author='Xintao Wang',
        author_email='xintao.wang@outlook.com',
        keywords='computer vision, restoration, super resolution',
        url='https://github.com/xinntao/BasicSR',
        include_package_data=True,
        packages=find_packages(exclude=('options', 'datasets', 'experiments', 'results', 'tb_logger', 'wandb')),
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
        ],
        license='Apache License 2.0',
        setup_requires=['setuptools', 'wheel'],
        install_requires=get_requirements(),
        ext_modules=ext_modules,
        zip_safe=False,
        **setup_kwargs
    )