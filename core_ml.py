import coremltools as ct
import urllib
import torchvision
import torch
import PIL
import time
import numpy as np


def main():
    # Load a pre-trained version of MobileNetV2 model.
    torch.hub.set_dir('./torch_hub_models')
    torch_model = torchvision.models.mobilenet_v2(pretrained=True)

    # Set the model in evaluation mode.
    torch_model.eval()

    # Trace the model with random data.
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(torch_model, example_input)
    out = traced_model(example_input)

    # Download class labels in ImageNetLabel.txt.
    label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    class_labels = urllib.request.urlopen(
        label_url).read().decode("utf-8").splitlines()
    # remove the first class which is background
    class_labels = class_labels[1:]
    assert len(class_labels) == 1000

    # Set the image scale and bias for input image preprocessing.
    scale = 1/(0.226*255.0)
    bias = [- 0.485/(0.229), - 0.456/(0.224), - 0.406/(0.225)]

    image_input = ct.ImageType(name="input_1",
                               shape=example_input.shape,
                               scale=scale, bias=bias)

    # Using image_input in the inputs parameter:
    # Convert to Core ML using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(class_labels),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    # # Save the converted model.
    # model.save("mobilenet.mlmodel")
    # # Print a confirmation message.
    # print('model converted and saved')

    # Load the test image and resize to 224, 224.
    img_path = "./images/daisy.jpeg"
    img = PIL.Image.open(img_path)
    img = img.resize([224, 224], PIL.Image.ANTIALIAS)

    # Get the protobuf spec of the model.
    spec = model.get_spec()
    for out in spec.description.output:
        if out.type.WhichOneof('Type') == "dictionaryType":
            coreml_dict_name = out.name
            break

    # Make a prediction with the Core ML version of the model.

    num_trials = 100
    inference_times = []

    print(f"Benchmarking inference: {num_trials} trials")
    with torch.no_grad():
        for i in range(num_trials):
            start_time = time.time()
            coreml_out_dict = model.predict({"input_1": img})
            end_time = time.time()
            inference_times.append(end_time - start_time)

    average_inference_time = sum(inference_times) / num_trials

    print("coreml predictions: ")
    print("top class label: ", coreml_out_dict["classLabel"])

    coreml_prob_dict = coreml_out_dict[coreml_dict_name]

    values_vector = np.array(list(coreml_prob_dict.values()))
    keys_vector = list(coreml_prob_dict.keys())
    top_3_indices_coreml = np.argsort(-values_vector)[:3]
    for i in range(3):
        idx = top_3_indices_coreml[i]
        score_value = values_vector[idx]
        class_id = keys_vector[idx]
        print("class name: {}, raw score value: {}".format(class_id, score_value))

    print(f"Average inference time: {average_inference_time * 1000:.2f} ms")


if __name__ == '__main__':
    main()
