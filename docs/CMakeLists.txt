
find_package(Sphinx REQUIRED)

add_custom_target(docs)

add_subdirectory(doxygen)
add_subdirectory(sphinx)
add_dependencies(docs nvimagecodec_doxygen_docs)
add_dependencies(docs nvimagecodec_sphinx_docs)
add_dependencies(nvimagecodec_sphinx_docs nvimagecodec_doxygen_docs)

