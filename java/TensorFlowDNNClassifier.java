package com.tensorflow.service;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class TensorFlowDNNClassifier {
	
	public static void main(String[] args) {
		
		int num_of_classes = 2;
		
		SavedModelBundle bundle = SavedModelBundle.load("C:\\model\\export\\1498915666", "serve");
		
		Session session = bundle.session();
		
		float[][] data = {
				{0, 0}
				, {0, 1}
				, {1, 0}
				, {1, 1}
		};
		
		Tensor inputTensor = Tensor.create(data);
		
		Tensor result = session.runner()
				.feed("ParseExample/ParseExample", inputTensor)
				.fetch("dnn/binary_logistic_head/predictions/probabilities")
				.run().get(0);
		
		float[][] vector = result.copyTo(new float[data.length][num_of_classes]); //e.g. 4x2
		
		for(int i=0;i<vector.length;i++){
			
			if(vector[i][0] > vector[i][1]){
				
				//then 0
				System.out.println(data[i][0]+" XOR "+data[i][1]+" would be -> 0 ("+100*vector[i][0]+"%)");
				
			}
			else{
				
				//then 1
				System.out.println(data[i][0]+" XOR "+data[i][1]+" would be -> 1 ("+100*vector[i][1]+"%)");
				
			}
			
		}
		
		bundle.close();
		
		
		
	}
	
}