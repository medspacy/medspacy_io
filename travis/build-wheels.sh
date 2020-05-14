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
  "${PYBIN}/pip" wheel /io/ -w /io/dist/
done

# Bundle external shared libraries into the wheels
for whl in /io/dist/*.whl; do
    if [[ $whl == /io/dist/${PROJECT_NAME}* ]]; then
      if [[ $whl != *none-any.whl ]]; then
#       if not an os-dependent build, the following repair will complain
        auditwheel repair "$whl" --plat $PLAT -w /io/dist/
      fi
    else
      rm $whl
    fi
done
echo 'current location: '
echo `pwd`

ls / -l
ls /io/ -l
ls /io/dist -l

# Install packages and test
for PYBIN in ${PYBINS[@]}; do
    PYBIN="/opt/python/${PYBIN}/bin"
    "${PYBIN}/pip" install ${PROJECT_NAME} --no-index -f /io/dist
    (cd "$HOME";ls -l;ls / -l ;cd ${PROJECT_NAME}/tests; python -m unittest)
done
chmod -R 777 /io/