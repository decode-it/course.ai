#include "gubg/mlp/Structure.hpp"
#include "gubg/data/Set.hpp"
#include "gubg/s11n.hpp"
#include "gubg/file/System.hpp"
#include "gubg/debug.hpp"
#include "gubg/std/optional.hpp"
#include "gubg/math/constants.hpp"
#include <string>
#include <iostream>
#include <cmath>
#include <random>
using namespace gubg;

std::mt19937 rng;

int main()
{
    S("");
    enum {
        LinearNeuron, LinearHiddenLayer,
        TanhNeuron, TanhHiddenLayer, TanhDeepNetwork,
        LeakyReLUNeuron, LeakyReLUHiddenLayer, LeakyReLUDeepNetwork,
        SoftPlusNeuron, SoftPlusHiddenLayer, SoftPlusDeepNetwork,
        Two33Two, Two555Two, Two515Two, Two55155TwoTanh, Two55155TwoLeakyReLU, Two55155TwoSoftPlus,
        LinearData, NoisyLinearData, SineData, NoisySineData,
        CircleData, CircleDataSame, TwoCircleData, Nr_};
    for (auto i = 0u; i < Nr_; ++i)
    {
        std::optional<mlp::Structure> mlp;
        std::optional<data::Set<double>> data;
        std::filesystem::path fn;
        switch (i)
        {
            case LinearNeuron:
                mlp.emplace(1);
                mlp->add_layer(neural::Transfer::Linear, 1, 0.0, 0.0);
                fn = "mlp.linear_neuron.naft";
                break;
            case LinearHiddenLayer:
                mlp.emplace(1);
                mlp->add_layer(neural::Transfer::Linear, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 1, 0.0, 0.0);
                fn = "mlp.linear_hidden_layer.naft";
                break;
            case TanhNeuron:
                mlp.emplace(1);
                mlp->add_layer(neural::Transfer::Tanh, 1, 0.0, 0.0);
                fn = "mlp.tanh_neuron.naft";
                break;
            case TanhHiddenLayer:
                mlp.emplace(1);
                mlp->add_layer(neural::Transfer::Tanh, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 1, 0.0, 0.0);
                fn = "mlp.tanh_hidden_layer.naft";
                break;
            case TanhDeepNetwork:
                mlp.emplace(1);
                mlp->add_layer(neural::Transfer::Tanh, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 1, 0.0, 0.0);
                fn = "mlp.tanh_deep_network.naft";
                break;
            case LeakyReLUNeuron:
                mlp.emplace(1);
                mlp->add_layer(neural::Transfer::LeakyReLU, 1, 0.0, 0.0);
                fn = "mlp.relu_neuron.naft";
                break;
            case LeakyReLUHiddenLayer:
                mlp.emplace(1);
                mlp->add_layer(neural::Transfer::LeakyReLU, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 1, 0.0, 0.0);
                fn = "mlp.relu_hidden_layer.naft";
                break;
            case LeakyReLUDeepNetwork:
                mlp.emplace(1);
                mlp->add_layer(neural::Transfer::LeakyReLU, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::LeakyReLU, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::LeakyReLU, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::LeakyReLU, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 1, 0.0, 0.0);
                fn = "mlp.relu_deep_network.naft";
                break;
            case SoftPlusNeuron:
                mlp.emplace(1);
                mlp->add_layer(neural::Transfer::SoftPlus, 1, 0.0, 0.0);
                fn = "mlp.softplus_neuron.naft";
                break;
            case SoftPlusHiddenLayer:
                mlp.emplace(1);
                mlp->add_layer(neural::Transfer::SoftPlus, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 1, 0.0, 0.0);
                fn = "mlp.softplus_hidden_layer.naft";
                break;
            case SoftPlusDeepNetwork:
                mlp.emplace(1);
                mlp->add_layer(neural::Transfer::SoftPlus, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::SoftPlus, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::SoftPlus, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::SoftPlus, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 1, 0.0, 0.0);
                fn = "mlp.softplus_deep_network.naft";
                break;
            case Two33Two:
                mlp.emplace(2);
                mlp->add_layer(neural::Transfer::Tanh, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 3, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 2, 0.0, 0.0);
                fn = "mlp.2332.naft";
                break;
            case Two555Two:
                mlp.emplace(2);
                mlp->add_layer(neural::Transfer::Tanh, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 2, 0.0, 0.0);
                fn = "mlp.25552.naft";
                break;
            case Two515Two:
                mlp.emplace(2);
                mlp->add_layer(neural::Transfer::Tanh, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 1, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 2, 0.0, 0.0);
                fn = "mlp.25152.naft";
                break;
            case Two55155TwoTanh:
                mlp.emplace(2);
                mlp->add_layer(neural::Transfer::Tanh, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 1, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Tanh, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 2, 0.0, 0.0);
                fn = "mlp.2551552.naft";
                break;
            case Two55155TwoLeakyReLU:
                mlp.emplace(2);
                mlp->add_layer(neural::Transfer::LeakyReLU, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::LeakyReLU, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::LeakyReLU, 1, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::LeakyReLU, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::LeakyReLU, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::Linear, 2, 0.0, 0.0);
                fn = "mlp.2551552_relu.naft";
                break;
            case Two55155TwoSoftPlus:
                mlp.emplace(2);
                mlp->add_layer(neural::Transfer::SoftPlus, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::SoftPlus, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::SoftPlus, 1, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::SoftPlus, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::SoftPlus, 5, 0.0, 0.0);
                mlp->add_layer(neural::Transfer::SoftPlus, 2, 0.0, 0.0);
                fn = "mlp.2551552_sp.naft";
                break;
            case LinearData:
                data.emplace();
                data->fields.emplace_back("input", 1);
                data->fields.emplace_back("output", 1);
                for (auto x = -3.0; x <= 3.0; x += 0.1)
                {
                    auto y = x/4.0+0.1;
                    auto &r = data->add_record();
                    r.add_data(x);
                    r.add_data(y);
                }
                fn = "data.linear.naft";
                break;
            case NoisyLinearData:
                {
                    std::normal_distribution<> normal(0, 0.1);
                    data.emplace();
                    data->fields.emplace_back("input", 1);
                    data->fields.emplace_back("output", 1);
                    for (auto x = -3.0; x <= 3.0; x += 0.2)
                    {
                        auto y = x/5.0+0.1+normal(rng);
                        auto &r = data->add_record();
                        r.add_data(x);
                        r.add_data(y);
                    }
                    fn = "data.noisy_linear.naft";
                }
                break;
            case SineData:
                data.emplace();
                data->fields.emplace_back("input", 1);
                data->fields.emplace_back("output", 1);
                for (auto x = -3.0; x <= 3.0; x += 0.1)
                {
                    auto y = std::sin(x);
                    auto &r = data->add_record();
                    r.add_data(x);
                    r.add_data(y);
                }
                fn = "data.sine.naft";
                break;
            case NoisySineData:
                {
                    std::normal_distribution<> normal(0, 0.1);
                    data.emplace();
                    data->fields.emplace_back("input", 1);
                    data->fields.emplace_back("output", 1);
                    for (auto x = -3.0; x <= 3.0; x += 0.2)
                    {
                        auto y = 0.7*std::sin(x)+normal(rng);
                        auto &r = data->add_record();
                        r.add_data(x);
                        r.add_data(y);
                    }
                    fn = "data.noisy_sine.naft";
                }
                break;
            case CircleData:
                data.emplace();
                data->fields.emplace_back("input", 2);
                data->fields.emplace_back("output", 2);
                for (auto angle = 0.0; angle <= gubg::math::tau; angle += gubg::math::tau/30)
                {
                    auto &r = data->add_record();
                    r.add_data(2.5*std::cos(angle), 0.8*std::sin(angle));
                    r.add_data(-std::sin(angle), std::cos(angle));
                }
                    fn = "data.circle.naft";
                break;
            case CircleDataSame:
                {
                    data.emplace();
                    data->fields.emplace_back("input", 2);
                    data->fields.emplace_back("output", 2);
                    std::uniform_real_distribution<> uniform(0.0, gubg::math::tau);
                    for (auto i = 0u; i < 100; ++i)
                    {
                        const auto angle = uniform(rng);
                        auto &r = data->add_record();
                        r.add_data(2.5*std::cos(angle), 0.8*std::sin(angle));
                        r.add_data(2.5*std::cos(angle), 0.8*std::sin(angle));
                    }
                    fn = "data.circle_same.naft";
                }
                break;
            case TwoCircleData:
                data.emplace();
                data->fields.emplace_back("input", 2);
                data->fields.emplace_back("output", 2);
                for (auto angle = 0.0; angle <= gubg::math::tau; angle += gubg::math::tau/30)
                {
                    {
                        auto &r = data->add_record();
                        r.add_data(2.5*std::cos(angle), 0.8*std::sin(angle));
                        r.add_data(-std::sin(angle), std::cos(angle));
                    }
                    {
                        auto &r = data->add_record();
                        r.add_data(1.25*std::cos(angle), 0.4*std::sin(angle));
                        r.add_data(std::sin(angle), -std::cos(angle));
                    }
                }
                    fn = "data.two_circle.naft";
                break;
            default:
                break;
        }

        std::filesystem::path dir = "data";
        if (!std::filesystem::exists(dir))
            std::filesystem::create_directory(dir);
        fn = dir / fn;

        std::cout << "Creating " << fn << " ... ";
        if (mlp)
            s11n::write_object_to_file(fn, ":mlp.Structure", *mlp);
        if (data)
            s11n::write_object_to_file(fn, ":data.Set", *data);
        std::cout << " done" << std::endl;
    }
    return 0;
}
