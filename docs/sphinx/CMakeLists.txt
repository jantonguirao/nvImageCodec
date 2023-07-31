

set(NVIMGCDCS_DOCS_TAG "public")

set(NVIMGCDCS_DOC_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}${PROJECT_VERSION_SUFFIX})

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/_build/html/index.html

    COMMAND ${CMAKE_MAKE_PROGRAM} html
    SPHINXOPTS=\"-t ${NVIMGCDCS_DOCS_TAG} -Q -Dversion=${NVIMGCDCS_DOC_VERSION} -Drelease=${NVIMGCDCS_DOC_VERSION}\"
    DEPENDS

    conf.py
    index.rst
    py_api.rst
    samples/python_samples.rst

    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(nvimagecodecs_sphinx_docs ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/_build/html/index.html)


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/_build/html 
    DESTINATION docs
    PATTERN ".buildinfo" EXCLUDE
  )

