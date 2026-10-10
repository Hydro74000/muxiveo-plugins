// SPDX-License-Identifier: LGPL-2.1-or-later
// Chargement privé du pilote système ; son absence laisse le secours CPU utilisable.
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
extern "C" PFN_vkGetInstanceProcAddr fel_vk_get_proc_addr()
{
    // Conserver le chargeur pendant toute la vie du module, initialisation thread-safe.
    static auto proc = []() -> PFN_vkGetInstanceProcAddr {
#ifdef _WIN32
        HMODULE lib = LoadLibraryW(L"vulkan-1.dll");
        return lib ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(lib,"vkGetInstanceProcAddr")) : nullptr;
#else
#ifdef __APPLE__
        void *lib = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
        if (!lib) lib = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
#else
        void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
        return lib ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib,"vkGetInstanceProcAddr")) : nullptr;
#endif
    }();
    return proc;
}
