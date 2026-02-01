add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor deeplearn)

add_executable(test_layer test_layer.cpp)
target_link_libraries(test_layer deeplearn)
