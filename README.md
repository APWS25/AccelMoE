# AccelMoE: Accelerated Mixture-of-Expert model 
AccelMoE is a project that optimizes a CPU-based mixture-of-experts architecture into GPU-based accelerated code. The project utilizes CUDA kernel programming to effectively execute computations on the GPU. The project was awarded **3rd Place** at the Accelerator Programming School competition.

> [!NOTE]
> This work is a project conducted as part of the Accelerator Programming School at Seoul National University.

## Optimization Overview
![Archeture](https://github.com/user-attachments/assets/8a59b9d1-f034-477d-92cd-a617eb86a734)

## Optimization Techniques
- GPU formatting using CUDA kernel programming
- Kernel fusion to combine Conv1D or Linear and ReLU operations
- CUDA streaming for efficient parallel processing
- Batch processing to maximize throughput
- Warp occupancy optimization

## Improved Performance
Achieved a 650× speedup when executed on the GPU.
### CPU version
```bash
Initializing inputs and parameters...Done!
Predicting sentiment...Done!
Elapsed time: 1.467701 (sec)
Throughput: 0.681338 (sentences/sec)
Finalizing...Done!
Saving outputs to ./data/outputs.bin...Done!
Validating...PASSED!
```

### GPU version
```bash
Initializing inputs and parameters...Done!
Predicting sentiment...Done!
Elapsed time: 0.074036 (sec)
Throughput: 432.224966 (sentences/sec)
Finalizing...Done!
Saving outputs to ./data/outputs.bin...Done!
Validating...PASSED!
```

## Contributors
|       |       |
| :---: | :---: |
|<img src="https://avatars.githubusercontent.com/u/89632139?v=4" style="width: 150px;">|<img src="https://avatars.githubusercontent.com/u/113115154?v=4" style="width: 150px;">|
|[Haeseung Jeon](https://github.com/JeonHaeseung)|[Suyeon Jo](https://github.com/Suyeonnie)|
|@Ewha Womans Univ.|@Myongji Univ.|
