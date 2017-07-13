package com.tensorflow.service;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class TensorFlowDnnRegressor {
	
	public static void main(String[] args) {
				
		SavedModelBundle bundle = SavedModelBundle.load("C:\\model\\export\\1499925271", "serve");
		
		Session session = bundle.session();
		
		float[][] data = {
				{(float) 0.83146961,(float) 0.70710678,(float) 0.55557023,(float) 0.38268343,(float) 0.19509032}
		};
		
		Tensor inputTensor = Tensor.create(data);
		
		Tensor result = session.runner()
				.feed("ParseExample/ParseExample", inputTensor)
				.fetch("dnn/regression_head/predictions/Identity")
				.run().get(0);
		
		System.out.println(result);
		
		float[] vector = result.copyTo(new float[data.length]);
		
		for(int i=0;i<vector.length;i++){
			
			System.out.println(vector[i]);
			
		}
		
		bundle.close();
		
	}

}