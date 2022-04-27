#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <sstream>
#include <getopt.h>

#include "host_common_util.hpp"
#include "profile_bnorm_fwd_impl.hpp"

using namespace std;

static struct option long_options[] = {{"inOutLengths", required_argument, nullptr, 'D'},
                                       {"half", no_argument, nullptr, '?'},
                                       {"double", no_argument, nullptr, '?'},
                                       {"int8", no_argument, nullptr, '?'},
                                       {"bf16", no_argument, nullptr, '?'},
                                       {"scales", required_argument, nullptr, 'S'},
                                       {"dumpout", required_argument, nullptr, 'o'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class BatchNormFwdProfilerArgs
{
    private:
    int option_index = 0;

    public:
    bool use_half   = false;
    bool use_double = false;
    bool use_int8   = false;
    bool use_bf16   = false;

    std::vector<size_t> inOutLengths;

    std::vector<float> scales;

    bool do_verification = false;
    bool do_dumpout      = false;

    bool saveMeanAndInvVariance;
    bool updateMovingAverage;

    int init_method;
    int nrepeat;

    BatchNormFwdProfilerArgs()  = default;
    ~BatchNormFwdProfilerArgs() = default;

    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout
            << "--inOutLengths or -D, comma separated list of input/output tensor dimension lengths"
            << std::endl;
        std::cout << "--half, use fp16 for the input and output tensor data types" << std::endl;
        std::cout << "--double, use fp64 for the input and output tensor data types" << std::endl;
        std::cout << "--int8, use int8 for the input and output tensor data types" << std::endl;
        std::cout << "--bf16, use bfloat16 for the input and output tensor data types" << std::endl;
        std::cout << "--scales or -S, comma separated two float values for alpha and beta"
                  << std::endl;
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the batch-norm result by "
                     "comparing with the host-based batch-norm"
                  << std::endl;
        std::cout
            << "--dumpout or -o, 1/0 to indicate where to save the batch-norm result to files "
               "for further analysis"
            << std::endl;
        std::cout << "Arg1 -- 1/0 to indicate whether to save the calculated mean and invVariance"
                  << std::endl;
        std::cout << "Arg2 -- 1/0 to indicate whether to update the moving average of the mean and "
                     "variance"
                  << std::endl;
        std::cout << "Arg3 -- init method used for bnScale and bnBias (0=no init, 1=single integer "
                     "value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
        std::cout << "Arg4 -- number of repeats to run the kernel" << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        using ck::host_common::getTypeValuesFromString;

        unsigned int ch;

        optind++; // to skip the "bnorm_fwd" module name

        while(1)
        {
            ch = getopt_long(argc, argv, "D:v:S:o:", long_options, &option_index);
            if(ch == -1)
                break;
            switch(ch)
            {
            case 'D':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                inOutLengths = getTypeValuesFromString<size_t>(optarg);
                break;
            case 'v':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                do_verification = static_cast<bool>(std::atoi(optarg));
                break;
            case 'S':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                scales = getTypeValuesFromString<float>(optarg);

                if(scales.size() != 2)
                    throw std::runtime_error("Invalid option format!");
                break;
            case 'o':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                do_dumpout = static_cast<bool>(std::atoi(optarg));
                break;
            case '?':
                if(std::string(long_options[option_index].name) == "half")
                    use_half = true;
                else if(std::string(long_options[option_index].name) == "double")
                    use_double = true;
                else if(std::string(long_options[option_index].name) == "int8")
                    use_int8 = true;
                else if(std::string(long_options[option_index].name) == "bf16")
                    use_bf16 = true;
                else if(std::string(long_options[option_index].name) == "help")
                {
                    show_usage(argv[0]);
                    return (-1);
                };
                break;

            default:
                show_usage(argv[0]);
                std::cerr << "Invalid cmd-line options!" << std::endl;
                return (-1);
            };
        };

        if(optind + 4 > argc)
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

        if(scales.empty())
        {
            scales.push_back(1.0f);
            scales.push_back(0.0f);
        };

        saveMeanAndInvVariance = std::atoi(argv[optind++]);
        updateMovingAverage    = std::atoi(argv[optind++]);

        init_method = std::atoi(argv[optind++]);
        nrepeat     = std::atoi(argv[optind]);

        if(do_verification && scales[1] != 0.0f && nrepeat > 0)
            throw std::runtime_error(
                "For verification, beta != 0.0f can only be used with nrepeat == 0");

        return (0);
    };

}; // end of class AppArgs

int profile_bnorm_fwd(int argc, char* argv[])
{
    using ck::profiler::profile_bnorm_fwd_impl;

    const double exponentialAverageFactor = 0.2;

    BatchNormFwdProfilerArgs args;

    if(args.processArgs(argc, argv) < 0)
        return (-1);

    int rank = args.inOutLengths.size();

    if(rank != 4)
    {
        throw std::runtime_error(
            "The input/out tensor lengths must have 4 dimensions for NHWC layout!");
    }

    // currently only NHWC layout and spatial batch-norm mode supported
    std::vector<size_t> scaleBiasMeanVarLengths = {args.inOutLengths[3]};

    if(args.use_half)
    {
        const double epsilon = 0.0001;

        profile_bnorm_fwd_impl<ck::half_t, float>(args.do_verification,
                                                  args.init_method,
                                                  args.do_dumpout,
                                                  args.nrepeat,
                                                  args.inOutLengths,
                                                  scaleBiasMeanVarLengths,
                                                  args.saveMeanAndInvVariance,
                                                  args.updateMovingAverage,
                                                  epsilon,
                                                  exponentialAverageFactor,
                                                  args.scales[0],
                                                  args.scales[1]);
    }
    else if(args.use_double)
    {
        const double epsilon = std::numeric_limits<double>::epsilon();

        profile_bnorm_fwd_impl<double, double>(args.do_verification,
                                               args.init_method,
                                               args.do_dumpout,
                                               args.nrepeat,
                                               args.inOutLengths,
                                               scaleBiasMeanVarLengths,
                                               args.saveMeanAndInvVariance,
                                               args.updateMovingAverage,
                                               epsilon,
                                               exponentialAverageFactor,
                                               args.scales[0],
                                               args.scales[1]);
    }
    else if(args.use_int8)
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        profile_bnorm_fwd_impl<int8_t, float>(args.do_verification,
                                              args.init_method,
                                              args.do_dumpout,
                                              args.nrepeat,
                                              args.inOutLengths,
                                              scaleBiasMeanVarLengths,
                                              args.saveMeanAndInvVariance,
                                              args.updateMovingAverage,
                                              epsilon,
                                              exponentialAverageFactor,
                                              args.scales[0],
                                              args.scales[1]);
    }
    else if(args.use_bf16)
    {
        const double epsilon = 0.0001;

        profile_bnorm_fwd_impl<ck::bhalf_t, float>(args.do_verification,
                                                   args.init_method,
                                                   args.do_dumpout,
                                                   args.nrepeat,
                                                   args.inOutLengths,
                                                   scaleBiasMeanVarLengths,
                                                   args.saveMeanAndInvVariance,
                                                   args.updateMovingAverage,
                                                   epsilon,
                                                   exponentialAverageFactor,
                                                   args.scales[0],
                                                   args.scales[1]);
    }
    else
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        profile_bnorm_fwd_impl<float, float>(args.do_verification,
                                             args.init_method,
                                             args.do_dumpout,
                                             args.nrepeat,
                                             args.inOutLengths,
                                             scaleBiasMeanVarLengths,
                                             args.saveMeanAndInvVariance,
                                             args.updateMovingAverage,
                                             epsilon,
                                             exponentialAverageFactor,
                                             args.scales[0],
                                             args.scales[1]);
    };

    return (0);
};
