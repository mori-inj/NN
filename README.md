# NN

* Tensorflow를 배우기 전, 직접 만든 Neural Network Library
* C++ 사용. 아래의 von에서 GUI요소를 제거하고 최적화 및 코드 리팩토링.
* von과 마찬가지로 신경망 모델이 layer 형태가 아니더라도 DAG기만 하면 학습 가능(Backpropagation 자체는 기존의 방식과 동일)
* layer사이의 가중치를 행렬로 표현해 병렬처리함으로써 얻을 수 있는 속도 상의 이점을 포기하고 표현 가능한 모델의 범위를 넓힘.
* Sigmoid, ReLU, PreLU(leaky ReLU) 등의 activation function 사용 가능
* 사용 예시
  ```c++
  
  //initialize fully connected, feedforward neural network
  FNN model;
  int layer[2] = {256, 256};
  model.add_input_layer(28*28);
  model.add_output_layer(10);
  for(int i=0; i<2; i++) {
      model.add_layer(layer[i], 
        [](LD x) -> LD{return PReLU(x);},
        [](LD x) -> LD{return deriv_PReLU(x);}
      );
  }
  model.add_all_weights();
  
  ...
  
  //train model
  for(int i=0; i<TRAIN_NUM; i++) {
      model.train(0.001, input_data_list, output_data_list);
  }
  
  ...
  
  //print result
  printf("train: %Lf (%Lf%%), test: %Lf (%Lf%%)\n",
      model.get_error(input_data_list, output_data_list), 
      model.get_precision(input_data_list, output_data_list)*100, 
      model.get_error(input_test_data_list, output_test_data_list), 
      model.get_precision(input_test_data_list, output_test_data_list)*100
  ); 
  ```
* MNIST data를 위 모델로 학습시킨 결과 92% 이상의 정확도를 얻음  
* 추가 예정  
  * 모델의 형태에 따라 layer구조인 경우 가중치를 행렬로 표현해 병렬화 가능하게끔  
  * convolution, pooling 구현(for cnn)  
  * cycle이 있는 모델에서도 사용할 수 있게끔(for rnn, lstm)  
