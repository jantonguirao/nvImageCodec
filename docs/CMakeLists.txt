
find_package(Sphinx REQUIRED)

add_custom_target(docs)

add_subdirectory(sphinx)
add_dependencies(docs nvimagecodecs_sphinx_docs)
# run pipeline

