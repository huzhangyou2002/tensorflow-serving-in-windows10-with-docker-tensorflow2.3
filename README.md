# tensorflow-serving-in-windows10-with-docker-tensorflow2.3

tensorflow-serving

#windows目录问题，以下指令在windows 10的power shell中可以执行，并成功返回
<br/>
docker run -t --rm -p 8501:8501 -v "E:/docker/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu:/models/half_plus_two" -e MODEL_NAME=half_plus_two tensorflow/serving
<br/>
#通过 python编写如下代码，可以成功获得预测结果
<br/>
import json
<br/>
import requests
<br/>
url = 'http://localhost:8501/v1/models/half_plus_two:predict'
<br/>
data = {"instances": [1.0, 2.0, 5.0]}
<br/>
r = requests.post(url, json.dumps(data))
<br/>
print(r)
<br/>
