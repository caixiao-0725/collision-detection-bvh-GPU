FILE(GLOB SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

FILE(GLOB HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh, ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# 生成链接库
add_library(Vector ${SRC_FILES} ${HEADER_FILES})

install(FILES ${HEADER_FILES}
    DESTINATION ${HEADER_INSTALL_PATH}/vector)
install(TARGETS ${TARGET_NAME}
    DESTINATION ${LIB_INSTALL_PATH})