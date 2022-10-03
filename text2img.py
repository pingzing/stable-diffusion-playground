import click
import datetime
from diffusers import StableDiffusionOnnxPipeline, DDIMScheduler
import numpy as np

@click.command()
@click.option("-p", "--prompt", required=True, type=str)
@click.option("-w", "--width", required=False, type=int, default=512)
@click.option("-h", "--height", required=False, type=int, default=512)
@click.option("-st", "--steps", required=False, type=int, default=25)
@click.option("-g", "--guidance-scale", required=False, type=float, default=7.5)
@click.option("-s", "--seed", required=False, type=int, default=None)
def run(
    prompt: str, 
    width: int, 
    height: int, 
    steps: int, 
    guidance_scale: float, 
    seed: int):

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="np")

    pipe = StableDiffusionOnnxPipeline.from_pretrained(
        "./stable_diffusion_onnx", 
        provider="DmlExecutionProvider",
        scheduler=scheduler
    )
    starttime = datetime.datetime.now()
    # print(starttime)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images)) # Disable the safety checker
    
    # Generate our own latents so that we can provide a seed.
    seed = np.random.randint(np.iinfo(np.int32).max) if seed is None else seed
    latents = get_latents_from_seed(seed, width, height)

    print(f"\nUsing a seed of {seed}")
    image = pipe(prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=guidance_scale, latents=latents).images[0]
    endtime = datetime.datetime.now()
    # print(endtime)
    imagetime = endtime.strftime("%Y%m%d%H%M%S")
    imagename = "output-" + imagetime + ".png"
    image.save(imagename)

def get_latents_from_seed(seed: int, width: int, height:int) -> np.ndarray:
    # 1 is batch size
    latents_shape = (1, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents

if __name__ == '__main__':
    run()
