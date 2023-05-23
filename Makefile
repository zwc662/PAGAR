VENV=neurips2023
# Recommended for universal compatibility ${CUDA_VERSION}=11.6.0
ifndef CUDA_VERSION
CUDA_VERSION=11.6.0
endif

venv:
	conda create -n ${VENV} python=3.6.9
	conda activate ${VENV}

cuda:
	conda install -c "nvidia/label/cuda-${CUDA_VERSION}" cuda-nvcc cuda-toolkit libcusparse-dev libcusolver-dev 
	export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
	sudo ln -s ${CONDA_PREFIX}/lib/libcudart.so /usr/lib/libcudart.so
	sudo ln -s ${CONDA_PREFIX}/lib/libcudart_static.a /usr/lib/libcudart.a
	sudo ln -s ${CONDA_PREFIX}/lib/libcurand.so /usr/lib/libcurand.so
	sudo ln -s ${CONDA_PREFIX}/lib/libcurand_static.a /usr/lib/libcurand.a
lib:
	python -m pip install -r requirements.txt
 