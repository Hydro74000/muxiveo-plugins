// Muxiveo — lecture / écriture de flux YUV4MPEG2 (y4m) en streaming.

#include "y4m.h"

#include <cstdlib>
#include <cstring>

static bool parse_colorspace(const std::string& c, FrameFormat& fmt)
{
    std::string base = c;
    int depth = 8;

    // suffixe de profondeur : 420p10, 422p12, 444p16…
    if (c.size() > 4 && c[3] == 'p' && c[4] >= '0' && c[4] <= '9')
    {
        base = c.substr(0, 3);
        depth = atoi(c.c_str() + 4);
        if (depth < 8 || depth > 16)
            return false;
    }

    if (base.compare(0, 3, "420") == 0)
    {
        fmt.sub_x = 2;
        fmt.sub_y = 2;
        if (c == "420mpeg2")
            fmt.siting = ChromaSiting::Left;
        else if (c == "420paldv")
            fmt.siting = ChromaSiting::TopLeft;
        else if (c == "420jpeg" || c == "420")
            fmt.siting = ChromaSiting::Center;
        else if (base != "420")
            return false;
    }
    else if (base == "422")
    {
        fmt.sub_x = 2;
        fmt.sub_y = 1;
        fmt.siting = ChromaSiting::Left;
    }
    else if (base == "444")
    {
        fmt.sub_x = 1;
        fmt.sub_y = 1;
        fmt.siting = ChromaSiting::Center;
    }
    else
    {
        return false;
    }

    fmt.bit_depth = depth;
    fmt.bytes_per_sample = depth > 8 ? 2 : 1;
    return true;
}

Y4mReader::Y4mReader(FILE* _fp) : fp(_fp)
{
}

bool Y4mReader::read_line(std::string& line, size_t max_len)
{
    line.clear();
    for (;;)
    {
        // Lecture d'un caractère ; taille contrôlée avant ajout au std::string.
        int ch = fgetc(fp); // flawfinder: ignore
        if (ch == EOF)
            return false;
        if (ch == '\n')
            return true;
        if (line.size() >= max_len)
            return false;
        line.push_back((char)ch);
    }
}

bool Y4mReader::read_header(std::string& error)
{
    std::string line;
    if (!read_line(line, 4096))
    {
        error = "flux y4m vide ou en-tête illisible";
        return false;
    }
    if (line.compare(0, 9, "YUV4MPEG2") != 0)
    {
        error = "signature YUV4MPEG2 absente (entrée y4m attendue)";
        return false;
    }

    // 420jpeg 8 bits par défaut (spécification y4m)
    parse_colorspace("420jpeg", fmt);

    size_t pos = 9;
    while (pos < line.size())
    {
        while (pos < line.size() && line[pos] == ' ')
            pos++;
        size_t end = line.find(' ', pos);
        if (end == std::string::npos)
            end = line.size();
        if (end == pos)
            break;
        std::string tok = line.substr(pos, end - pos);
        pos = end;

        switch (tok[0])
        {
        case 'W':
            fmt.width = atoi(tok.c_str() + 1);
            break;
        case 'H':
            fmt.height = atoi(tok.c_str() + 1);
            break;
        case 'F':
        {
            long long num = 0, den = 0;
            if (sscanf(tok.c_str() + 1, "%lld:%lld", &num, &den) != 2 || num <= 0 || den <= 0)
            {
                error = "cadence y4m invalide : " + tok;
                return false;
            }
            fmt.fps_num = num;
            fmt.fps_den = den;
            break;
        }
        case 'I':
            fmt.interlace = tok.size() > 1 ? tok[1] : '?';
            tokens.push_back(tok);
            break;
        case 'C':
            if (!parse_colorspace(tok.substr(1), fmt))
            {
                error = "espace colorimétrique y4m non supporté : " + tok.substr(1);
                return false;
            }
            tokens.push_back(tok);
            break;
        case 'X':
            if (tok == "XCOLORRANGE=LIMITED")
                fmt.range = ColorRange::Limited;
            else if (tok == "XCOLORRANGE=FULL")
                fmt.range = ColorRange::Full;
            tokens.push_back(tok);
            break;
        default:
            tokens.push_back(tok);
            break;
        }
    }

    if (fmt.width <= 0 || fmt.height <= 0)
    {
        error = "dimensions y4m absentes ou invalides";
        return false;
    }
    if (fmt.fps_num <= 0)
    {
        error = "cadence y4m absente (jeton F)";
        return false;
    }
    return true;
}

int Y4mReader::read_frame(uint8_t* dst, std::string& error)
{
    std::string line;
    if (!read_line(line, 1024))
    {
        if (line.empty() && feof(fp))
            return 0;
        error = "en-tête de trame y4m tronqué";
        return -1;
    }
    if (line.compare(0, 5, "FRAME") != 0)
    {
        error = "marqueur FRAME attendu, reçu : " + line.substr(0, 32);
        return -1;
    }

    const size_t size = fmt.frame_bytes();
    size_t got = 0;
    while (got < size)
    {
        size_t n = fread(dst + got, 1, size - got, fp);
        if (n == 0)
        {
            error = "trame y4m tronquée";
            return -1;
        }
        got += n;
    }
    return 1;
}

Y4mWriter::Y4mWriter(FILE* _fp) : fp(_fp)
{
}

bool Y4mWriter::write_header(const FrameFormat& fmt, const std::vector<std::string>& passthrough_tokens)
{
    std::string header = "YUV4MPEG2 W" + std::to_string(fmt.width) + " H" + std::to_string(fmt.height)
                         + " F" + std::to_string(fmt.fps_num) + ":" + std::to_string(fmt.fps_den);
    for (const std::string& tok : passthrough_tokens)
        header += " " + tok;
    header += "\n";
    return fwrite(header.data(), 1, header.size(), fp) == header.size();
}

bool Y4mWriter::write_frame(const uint8_t* data, size_t size)
{
    static const char marker[] = "FRAME\n";
    if (fwrite(marker, 1, 6, fp) != 6)
        return false;
    return fwrite(data, 1, size, fp) == size;
}

bool Y4mWriter::flush()
{
    return fflush(fp) == 0;
}
