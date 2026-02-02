add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor deeplearn)

add_executable(test_layer test_layer.cpp)
target_link_libraries(test_layer deeplearn)

add_executable(test_optim test_optim.cpp)
target_link_libraries(test_optim deeplearn)

add_executable(example_sgd_training example_sgd_training.cpp)
target_link_libraries(example_sgd_training deeplearn)


