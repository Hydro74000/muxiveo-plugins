// Garde-fou d'exécution (voir gpu_watchdog.h).
#include "gpu_watchdog.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include "gpu.h"

namespace watchdog
{
namespace
{
using Clock = std::chrono::steady_clock;

struct Probe
{
    const ncnn::VulkanDevice* dev = nullptr;
    VkFence fence = VK_NULL_HANDLE;
};

constexpr int GRACE_SECONDS = 5;

std::mutex m;
std::condition_variable cv;
std::thread th;
bool running = false;
int stall_s = 0, gpu_code = 3;
std::vector<Probe> probes;
std::atomic<int64_t> last_progress{0};  // nanosecondes (horloge monotone)
std::atomic<int> io_waiters{0};
std::atomic<bool> fatal_set{false};
std::atomic<int> fatal_code{0};
std::atomic<int64_t> fatal_deadline{0};

int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }

void loop()
{
    std::unique_lock<std::mutex> lk(m);
    int64_t next_probe = 0;
    while (running)
    {
        cv.wait_for(lk, std::chrono::milliseconds(250));
        if (!running)
            break;
        const int64_t t = now_ns();
        if (fatal_set)
        {
            if (t >= fatal_deadline)
            {
                fprintf(stderr, "error: arrêt forcé après une erreur GPU (fils bloqués dans le pilote)\n");
                fflush(stderr);
                std::_Exit(fatal_code);
            }
            continue;
        }
        if (t >= next_probe)
        {
            next_probe = t + 1000000000LL;
            for (const Probe& p : probes)
                if (p.fence != VK_NULL_HANDLE && ncnn::vkGetFenceStatus(p.dev->vkdevice(), p.fence) == VK_ERROR_DEVICE_LOST)
                {
                    lk.unlock();
                    fatal(gpu_code, "périphérique GPU perdu (VK_ERROR_DEVICE_LOST)");
                    lk.lock();
                    break;
                }
        }
        if (stall_s > 0 && !fatal_set && io_waiters.load() == 0 && t - last_progress.load() > (int64_t)stall_s * 1000000000LL)
        {
            char msg[128];
            snprintf(msg, sizeof msg, "blocage : aucune trame traitée depuis %d s", stall_s);
            lk.unlock();
            fatal(gpu_code, msg);
            lk.lock();
        }
    }
}
}  // namespace

void start(int stall_seconds, int gpu_exit_code)
{
    std::lock_guard<std::mutex> lk(m);
    if (running)
        return;
    stall_s = std::max(0, stall_seconds);
    gpu_code = gpu_exit_code;
    last_progress = now_ns();
    running = true;
    th = std::thread(loop);
}

void watch_device(const ncnn::VulkanDevice* dev)
{
    if (!dev)
        return;
    std::lock_guard<std::mutex> lk(m);
    for (const Probe& p : probes)
        if (p.dev == dev)
            return;
    Probe p;
    p.dev = dev;
    VkFenceCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (ncnn::vkCreateFence(dev->vkdevice(), &info, 0, &p.fence) != VK_SUCCESS)
        p.fence = VK_NULL_HANDLE;
    probes.push_back(p);
}

void stop()
{
    {
        std::lock_guard<std::mutex> lk(m);
        if (!running)
            return;
        running = false;
    }
    cv.notify_all();
    th.join();
    std::lock_guard<std::mutex> lk(m);
    for (Probe& p : probes)
        if (p.fence != VK_NULL_HANDLE)
            ncnn::vkDestroyFence(p.dev->vkdevice(), p.fence, 0);
    probes.clear();
}

void progress() { last_progress = now_ns(); }

void io_begin() { io_waiters++; }

void io_end()
{
    io_waiters--;
    last_progress = now_ns();
}

void fatal(int exit_code, const char* message)
{
    bool expected = false;
    if (!fatal_set.compare_exchange_strong(expected, true))
        return;
    fatal_code = exit_code;
    fatal_deadline = now_ns() + (int64_t)GRACE_SECONDS * 1000000000LL;
    if (message && *message)
    {
        fprintf(stderr, "error: %s\n", message);
        fflush(stderr);
    }
}

bool failed() { return fatal_set; }
}  // namespace watchdog
