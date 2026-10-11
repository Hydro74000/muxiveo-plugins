// Vérification CPU des branches de caractéristiques avant toute mutualisation GPU.
#ifndef MUXIVEO_HEAD_SHARING_H
#define MUXIVEO_HEAD_SHARING_H

#include <cstddef>
#include <string>
#include <vector>

class HeadSharingProof
{
public:
    explicit HeadSharingProof(const std::string& param);
    bool next_load(int& count, int& type) const;
    void observe(int count, int type, const void* data, size_t bytes);
    bool equivalent() const;

private:
    struct Load { int count; int type; int side; };
    std::vector<Load> loads;
    std::vector<std::vector<unsigned char>> heads[2];
    size_t cursor = 0;
    bool valid = false;
};

#endif
