    // MUXIVEO_EXTERNAL_MEMORY_EXPORT : export de mémoire Vulkan vers CUDA (plugin TensorRT de muxiveo-rife).
    if (info.support_VK_KHR_external_memory())
    {
        uint32_t muxiveo_ext_count = 0;
        vkEnumerateDeviceExtensionProperties(info.physicalDevice(), 0, &muxiveo_ext_count, 0);
        std::vector<VkExtensionProperties> muxiveo_ext_props(muxiveo_ext_count);
        vkEnumerateDeviceExtensionProperties(info.physicalDevice(), 0, &muxiveo_ext_count, muxiveo_ext_props.data());
        for (uint32_t muxiveo_k = 0; muxiveo_k < muxiveo_ext_count; muxiveo_k++)
        {
#if _WIN32
            if (strcmp(muxiveo_ext_props[muxiveo_k].extensionName, "VK_KHR_external_memory_win32") == 0)
                enabledExtensions.push_back("VK_KHR_external_memory_win32");
#else
            if (strcmp(muxiveo_ext_props[muxiveo_k].extensionName, "VK_KHR_external_memory_fd") == 0)
                enabledExtensions.push_back("VK_KHR_external_memory_fd");
#endif
        }
    }
