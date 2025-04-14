from setuptools import setup
import os
from glob import glob

package_name = 'ai_tools'

setup(
    name=package_name,
    version='0.0.7',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Only include dataset if files exist
        (os.path.join('share', package_name, 'dataset'), glob('dataset/*')),
    ],
    install_requires=['setuptools', 'numpy', 'pandas', 'scikit-learn'],
    zip_safe=True,
    maintainer='Bogdan ABAZA',
    maintainer_email='bogdan.abaza@upb.ro',
    description='Tools for AI-based covariance prediction in robotic navigation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ai_covariance_node = ai_tools.ai_covariance_node:main',
            # Include updater if it exists
            'ai_covariance_updater = ai_tools.ai_covariance_updater:main',
        ],
    },
)