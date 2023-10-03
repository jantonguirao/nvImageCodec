

set(NVIMGCODEC_DOCS_TAG "public")

set(NVIMGCODEC_DOC_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}${PROJECT_VERSION_SUFFIX})

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/_build/html/index.html

    COMMAND ${CMAKE_MAKE_PROGRAM} html
    SPHINXOPTS=\"-t ${NVIMGCODEC_DOCS_TAG} -Q -Dversion=${NVIMGCODEC_DOC_VERSION} -Drelease=${NVIMGCODEC_DOC_VERSION}\"
    DEPENDS

    conf.py
    index.rst
    py_api.rst
    samples/python_samples.rst

    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(nvimagecodec_sphinx_docs DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/_build/html/index.html)


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/_build/html 
    DESTINATION docs
    PATTERN ".buildinfo" EXCLUDE
  )

