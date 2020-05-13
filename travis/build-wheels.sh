#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y atlas-devel
ARGS=("$@")

PLAT=${ARGS[0]}
PROJECT_NAME=${ARGS[1]}
PYBINS=${ARGS[@]:2}
# Compile wheels
echo "${PYBINS[@]}"
for PYBIN in ${PYBINS[@]};do
  PYBIN="/opt/python/${PYBIN}/bin"
  echo "${PYBIN}"
  "${PYBIN}/pip" install -q -r /io/dev-requirements.txt
  "${PYBIN}/python" -m spacy download en
  "${PYBIN}/pip" wheel /io/ -w /io/wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in /io/wheelhouse/*.whl; do
    if [[ $whl == /io/wheelhouse/${PROJECT_NAME}* ]]; then
      if [[ $whl != *none-any.whl ]]; then
        auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
      fi
    else
      rm $whl
    fi
done

echo `pwd`

ls / -l

ls /io/wheelhouse -l

# Install packages and test
for PYBIN in ${PYBINS[@]}; do
    PYBIN="/opt/python/${PYBIN}/bin"
    "${PYBIN}/pip" install ${PROJECT_NAME} --no-index -f /io/wheelhouse
    (cd "$HOME";ls -l; "${PYBIN}/nosetests" ${PROJECT_NAME})
done