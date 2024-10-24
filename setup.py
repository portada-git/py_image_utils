from setuptools import setup

setup(name='py_image_utils',
    version='0.0.2',
    description='image utils for PortADa project',
    author='PortADa team',
    author_email='jcbportada@gmail.com',
    license='MIT',
    url="https://github.com/portada-git/py_image_utils",
    packages=['utils'],
    py_modules=['image_utilities_cv'],
    install_requires=[
	'opencv-python',
	'matplotlib',
	'numpy',
    ],
    python_requires='>=3.9',
    zip_safe=False)
