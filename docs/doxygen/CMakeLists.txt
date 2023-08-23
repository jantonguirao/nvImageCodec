find_package(Doxygen REQUIRED)

set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/xml/index.xml)

add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/nvimgcodecs.h
    COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    MAIN_DEPENDENCY Doxyfile
    COMMENT "Generating nvImageCodecs docs")

add_custom_target(nvimagecodecs_doxygen_docs DEPENDS ${DOXYGEN_INDEX_FILE})
