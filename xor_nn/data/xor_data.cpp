#include "xor_data.h"

std::vector<Sample> get_xor_data()
{
    return {
        { {0.0, 0.0}, {0.0} },
        { {0.0, 1.0}, {1.0} },
        { {1.0, 0.0}, {1.0} },
        { {1.0, 1.0}, {0.0} },
    };
}
