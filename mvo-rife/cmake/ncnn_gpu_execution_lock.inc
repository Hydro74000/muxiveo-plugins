    // Muxiveo (MUXIVEO_GPU_EXECUTION_SERIAL) : un seul lot de commandes en exécution à la fois dans le processus,
    // de la soumission à la fin de l'attente (voir cmake/patch_ncnn.cmake).
    static std::mutex muxiveo_gpu_execution;
    std::unique_lock<std::mutex> muxiveo_gpu_lock(muxiveo_gpu_execution);

