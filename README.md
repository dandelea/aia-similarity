# 05-Similitud de textos
Aplicación para la asignatura AIA (Aplicaciones de Inteligencia Artificial) del Máster de Ingeniería Informática de la Universidad de Sevilla.

## Instalación
* Actualizar las bases de datos de repositorios
 * <code>sudo apt-get update</code>
* Instalar Python 3.4:
 * <code>sudo apt-get install python3.4-dev</code>
* Instalar pip:
 * <code>cd ~</code>
 * <code>wget https://bootstrap.pypa.io/get-pip.py</code>
 * <code>sudo python3 get-pip.py</code>
* Instalar numpy, scipy y scikit
 * <code>pip install numpy</code>
 * <code>sudo apt-get install build-essential python3-dev python3-setuptools python3-numpy python3-scipy libatlas-dev libatlas3gf-base</code>
 * <code>sudo pip3 install -U scikit-learn</code>
* Instalar pytables y hdf5
 * <code>cd ~</code>
 * <code>sudo wget https://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.16.tar.gz</code>
 * <code>tar xzvf hdf5-1.8.16.tar.gz</code>
 * <code>cd hdf5-1.8.16</code>
 * <code>./configure</code>
 * <code>sudo make install</code>
 * <code>export HDF5_DIR=~/hdf5</code>
 * <code>sudo apt-get build-dep python-tables</code>
 * <code>sudo pip install tables</code>
* Instalar Matplotlib
 * <code>cd ~</code>
 * <code>git clone git://github.com/matplotlib/matplotlib.git</code>
 * <code>cd matplotlib</code>
 * <code>sudo python3 setup.py install</code>
* Instalar NLTK
 * <code>sudo pip3 install -U nltk</code>