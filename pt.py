import torch
import torchvision
import time
import argparse
from torchvision import transforms
from PIL import Image


def main(arguments):
    use_mps = arguments.mps

    if use_mps:
        # Check that MPS (Metal Performance Shaders) is available
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                      "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                      "and/or you do not have an MPS-enabled device on this machine.")
            raise SystemExit(0)

        print("Using MPS (GPU acceleration) for inference.")

    # Load model from cache or download and cache if needed
    torch.hub.set_dir('./torch_hub_models')
    print("Loading model...")
    model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # Set the model to evaluation mode
    model.eval()

    filename = "./images/dog.jpeg"

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if use_mps:
        mps_device = torch.device("mps")
        model.to(mps_device)
        input_batch = input_batch.to(mps_device)

    num_trials = 100
    inference_times = []

    print(f"Benchmarking inference: {num_trials} trials")
    with torch.no_grad():
        for i in range(num_trials):
            start_time = time.time()
            output = model(input_batch)
            end_time = time.time()
            inference_times.append(end_time - start_time)

    average_inference_time = sum(inference_times) / num_trials

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    with open("./torch_hub_models/imagenet_classes.txt", "r") as f:
        categories = [s.strip().capitalize() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    print("Top 5 categories:")
    for i in range(top5_prob.size(0)):
        print(f"{categories[top5_catid[i]]} : {top5_prob[i].item():.5f}")

    print(f"Average inference time: {average_inference_time * 1000:.2f} ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mps', action='store_true')
    args = parser.parse_args()
    main(args)
