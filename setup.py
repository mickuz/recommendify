from setuptools import setup

setup(name='Recommendify',
      version='0.1',
      description='Spotify Recommender System',
      author='Michal Kuzniewicz',
      author_email='michal.kuzniewicz@tuta.io',
      url='https://github.com/mickuz/recommendify',
      packages=['src'],
      install_requires=[
            'spotipy',
            'PyYAML',
            'pandas',
            'numpy',
            'scikit-learn'
      ],
      extras_require={
            'plotting': ['matplotlib', 'seaborn', 'jupyter'],
            'optimization': ['xgboost', 'hyperopt']
      },
      setup_requires=[
            'flake8',
            'pytest-runner'
      ],
      tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-mock'
      ],
      entry_points={
            'console_scripts': [
                  'fetch-data=src.data.raw_dataset:main',
                  'process-data=src.features.processed_dataset:main',
                  'train=src.models.train_model:main',
                  'predict=src.main:main'
            ]
      },
      package_data={
            'src': ['configs/config.yaml']
      })
