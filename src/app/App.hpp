#ifndef HEADER_app_App_hpp_ALREADY_INCLUDED
#define HEADER_app_App_hpp_ALREADY_INCLUDED

#include "app/Draw.hpp"
#include "app/Util.hpp"
#include "gubg/mlp/Structure.hpp"
#include "gubg/mlp/Parameters.hpp"
#include "gubg/neural/Simulator.hpp"
#include "gubg/neural/setup.hpp"
#include "gubg/neural/Trainer.hpp"
#include "gubg/data/Set.hpp"
#include "gubg/s11n.hpp"
#include "gubg/mss.hpp"
#include "gubg/imgui/SelectFile.hpp"
#include "gubg/imgui/CStringMgr.hpp"
#include "imgui.h"
#include "imgui-SFML.h"
#include "SFML/Graphics.hpp"
#include "gubg/std/filesystem.hpp"
#include "gubg/std/optional.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <list>
#include <vector>
#include <cmath>
#include <random>

namespace app { 

    using DataSet = gubg::data::Set<double>;

    class App
    {
    public:
        void run()
        {
            S("");

            load_font_();

            auto desktop_mode = sf::VideoMode::getDesktopMode();
            if (false)
            {
                desktop_mode.width /= 2;
                desktop_mode.height /= 2;
            }
            sf::RenderWindow window(desktop_mode, "");

            const auto window_size = window.getSize();
            L(C(window_size.x)C(window_size.y));
            io_.setup("", window_size.x, window_size.y, 0.0, sf::Color(0, 20, 0), font_);

            window.setVerticalSyncEnabled(true);
            ImGui::SFML::Init(window);

            {
                const bool is_hires = sf::VideoMode::getDesktopMode().width >= 3000;
                if (is_hires)
                    ImGui::GetIO().FontGlobalScale = 2.0;
                else
                    ImGui::GetIO().FontGlobalScale = 1.0;
            }

            window.resetGLStates(); // call it if you only draw ImGui. Otherwise not needed.

            sf::Clock clock;

            //Process events until windows closes
            while (window.isOpen()) {
                sf::Event event;
                while (window.pollEvent(event)) {
                    ImGui::SFML::ProcessEvent(event);

                    if (event.type == sf::Event::Closed) {
                        window.close();
                    }
                }

                ImGui::SFML::Update(window, clock.restart());

                {
                    ImGui::Begin("Neural Network: Multi-Layer Perceptron");

                    if (!imgui_mainloop_())
                    {
                        std::cout << "Error: ImGui mainloop failed, we will stop ASAP" << std::endl;
                        window.close();
                    }

                    ImGui::End();
                }

                const auto bg_color = (model_ ? sf::Color(0, 128, 128) : sf::Color(0, 0, 0));
                window.clear(bg_color);
                if (io_.valid)
                    io_.draw(window);
                ImGui::SFML::Render(window);
                window.display();
            }

            ImGui::SFML::Shutdown();
        }

    private:
        bool imgui_mainloop_()
        {
            MSS_BEGIN(bool);

            cstr_.reset();

            //Print the error, if any
            {
                const auto &str = error_.str();
                if (!str.empty())
                    ImGui::Text("Error: %s", str.c_str());
            }

            MSS(load_model_());
            if (model_)
            {
                MSS(display_mlp_structure_());
                MSS(display_weight_randomization_());
                MSS(simulate_mlp_());
            }

            ImGui::Separator();

            MSS(load_data_());
            if (dataset_)
            {
                MSS(display_data_());
            }

            ImGui::Separator();
            if (model_ && dataset_)
            {
                MSS(display_cost_weight_decay_());
                MSS(display_cost_());
                MSS(learn_model_());
            }

            MSS_END();
        }
        bool load_model_()
        {
            MSS_BEGIN(bool);
            const bool setup_default_structure = (false && structure_fn_.empty());
            if (setup_default_structure || gubg::imgui::select_file("Load model", structure_fn_))
            {
                if (setup_default_structure)
                    structure_fn_ = "data/mlp.tanh_neuron.naft";
                std::cout << "Selected structure file " << structure_fn_ << std::endl;
                learn_.reset();
                model_.reset();
                error_.str("");
                gubg::mlp::Structure structure;
                MSS(gubg::s11n::read_object_from_file(structure_fn_, ":mlp.Structure", structure), error_ << "Could not read MLP structure from " << structure_fn_.string());
                model_.emplace();
                model_->structure = structure;
                model_->parameters.setup_from(model_->structure);
            }
            ImGui::SameLine();
            ImGui::Text(structure_fn_.string().c_str());
            MSS_END();
        }
        bool load_data_()
        {
            MSS_BEGIN(bool);
            const bool setup_default_data = (false && data_fn_.empty());
            if (setup_default_data || gubg::imgui::select_file("Load data", data_fn_))
            {
                if (setup_default_data)
                    data_fn_ = "data/data.noisy_sine.naft";
                std::cout << "Selected data file " << data_fn_ << std::endl;
                learn_.reset();
                dataset_.reset();
                error_.str("");
                DataSet dataset;
                MSS(gubg::s11n::read_object_from_file(data_fn_, ":data.Set", dataset), error_ << "Could not read dataset from " << data_fn_.string());
                dataset_.emplace(dataset);
            }
            ImGui::SameLine();
            ImGui::Text(data_fn_.string().c_str());
            MSS_END();
        }
        bool display_mlp_structure_()
        {
            MSS_BEGIN(bool);

            MSS(!!model_);
            auto &model = *model_;

            auto &simulator = goc_simulator_();

            //Display network structure
            {
                ImGui::Text("Nr inputs: %d, nr weights: %d", model.structure.nr_inputs, simulator.nr_weights());
                auto &s = model.structure;
                for (auto lix = 0u; lix < s.layers.size(); ++lix)
                {
                    ImGui::Text(cstr_("Layer ", lix));
                    ImGui::SameLine();
                    auto &layer = s.layers[lix];
                    for (auto nix = 0u; nix < layer.neurons.size(); ++nix)
                    {
                        if (ImGui::RadioButton(cstr_("L", lix, "N", nix), (model.lix == lix) && (model.nix == nix)))
                        {
                            model.lix = lix;
                            model.nix = nix;
                        }
                        ImGui::SameLine();
                    }
                    for (auto &n: layer.neurons)
                        n.weight_stddev = model.weights_stddev;
                    for (auto &n: layer.neurons)
                        n.bias_stddev = model.weights_stddev;
                    ImGui::NewLine();
                    ImGui::Separator();
                }
            }

            const auto select_exhaustive_params = !!learn_ && learn_->algo == Algo::Exhaustive && !learn_->do_learn;

            //Display selected neuron
            ImGui::Text(cstr_("Selected neuron: layer ", model.lix, " neuron ", model.nix));
            {
                auto &neuron =  model.parameters.layers[model.lix].neurons[model.nix];
                for (auto wix = 0u; wix < neuron.weights.size(); ++wix)
                {
                    float weight = neuron.weights[wix];
                    ImGui::SliderFloat(cstr_("Weight ", wix), &weight, -3.0, 3.0);
                    neuron.weights[wix] = weight;
                    if (select_exhaustive_params)
                    {
                        ImGui::SameLine();
                        ImGui::Checkbox(cstr_("Exhaustive", wix), &learn_->exhaustive_map[ParamIX(model.lix, model.nix, wix)]);
                    }
                }
                {
                    float bias = neuron.bias;
                    ImGui::SliderFloat(cstr_("Bias"), &bias, -3.0, 3.0);
                    neuron.bias = bias;
                    if (select_exhaustive_params)
                    {
                        ImGui::SameLine();
                        ImGui::Checkbox("Exhaustive bias", &learn_->exhaustive_map[ParamIX(model.lix, model.nix, -1)]);
                    }
                }
                {
                    auto &neuron = model.structure.layers[model.lix].neurons[model.nix];
                    ImGui::Text("Transfer function: %s", to_str(neuron.transfer));
                }
            }
            ImGui::Separator();

            MSS_END();
        }
        bool display_cost_weight_decay_()
        {
            MSS_BEGIN(bool);

            MSS(!!model_);
            auto &model = *model_;

            {
                model.cost_stddev = std::max(model.cost_stddev, 0.01);
                float cost_stddev = model.cost_stddev;
                if (ImGui::SliderFloat("Output stddev", &cost_stddev, 0.01, 0.5))
                {
                    model.cost_stddev = std::max<float>(cost_stddev, 0.01);
                    model.simulator.reset();
                }
            }
            {
                model.weights_stddev = std::max(model.weights_stddev, 0.1);
                float weights_stddev = model.weights_stddev;
                if (ImGui::SliderFloat("Weights stddev", &weights_stddev, 0.1, 10.0))
                    model.weights_stddev = weights_stddev;
            }
            ImGui::Separator();

            MSS_END();
        }
        bool display_weight_randomization_()
        {
            MSS_BEGIN(bool);

            MSS(!!model_);
            auto &model = *model_;

            {
                float v = model.randomize_weights_stddev;
                ImGui::SliderFloat("Rnd stddev", &v, 0.001, 1.0);
                model.randomize_weights_stddev = v;
            }
            if (ImGui::Button("Randomize absolute"))
            {
                if (learn_)
                {
                    learn_->scg_do_init = true;
                    learn_->adam_do_init = true;
                }
                std::normal_distribution<double> gaussian(0.0, model.randomize_weights_stddev);
                for (auto &l: model.parameters.layers)
                    for (auto &n: l.neurons)
                    {
                        for (auto &w: n.weights)
                            w = gaussian(rng_);
                        n.bias = gaussian(rng_);
                    }
            }
            ImGui::SameLine();
            if (ImGui::Button("Randomize relative"))
            {
                if (learn_)
                {
                    learn_->scg_do_init = true;
                    learn_->adam_do_init = true;
                }
                std::normal_distribution<double> gaussian(0.0, model.randomize_weights_stddev);
                for (auto &l: model.parameters.layers)
                    for (auto &n: l.neurons)
                    {
                        for (auto &w: n.weights)
                            w += gaussian(rng_);
                        n.bias = gaussian(rng_);
                    }
            }
            ImGui::Separator();
            MSS_END();
        }
        bool simulate_mlp_()
        {
            MSS_BEGIN(bool);

            MSS(!!model_);
            auto &model = *model_;

            auto &simulator = goc_simulator_();
            auto &wnd = io_.goc();
            Transform t(wnd, 3,1);
            io_.line(1, sf::Color(30, 30, 0), [&](auto &line){ line.point(t(-3.0,0.0)).point(t(3.0,0.0)); });
            io_.line(1, sf::Color(30, 30, 0), [&](auto &line){ line.point(t(0.0,-1.0)).point(t(0.0,1.0)); });

            ImGui::SliderFloat("Weight scale", &model.weight_scale, -1.5, 1.5);
            ImGui::SameLine();
            if (ImGui::Button("Apply"))
            {
                auto multiply_weights = [&](auto &neuron, unsigned int, unsigned int){
                    for (auto &w: neuron.weights)
                        w *= model.weight_scale;
                    neuron.bias *= model.weight_scale;
                };
                model.parameters.each_neuron(multiply_weights);
            }

            MSS(gubg::neural::setup(model.weights, model.parameters));

            if (false) {}
            else if (model.structure.nr_inputs == 1 && model.structure.nr_outputs() == 1)
            {
                auto draw_io = [&](auto &line){
                    for (auto x = -3.0; x <= 3.0; x += 0.01)
                    {
                        model.states[model.input] = x;
                        simulator.forward(model.states.data(), model.weights.data());
                        const auto y = model.states[model.output];
                        line.point(t(x, y));
                    }
                };
                io_.line(1, sf::Color::Red, draw_io);
            }
            else if (model.structure.nr_inputs == 2 && model.structure.nr_outputs() == 2)
            {
                for (auto x0 = -3.0; x0 <= 3.0; x0 += 0.1)
                {
                    for (auto x1 = -1.0; x1 <= 1.0; x1 += 0.1)
                    {
                        model.states[model.input] = x0;
                        model.states[model.input+1] = x1;
                        simulator.forward(model.states.data(), model.weights.data());
                        const auto y0 = model.states[model.output];
                        const auto y1 = model.states[model.output+1];
                        io_.arrow(sf::Color::Red, t(x0,x1), sf::Vector2f(y0,y1)*10.0f);
                    }
                }
            }

            MSS_END();
        }
        bool display_data_()
        {
            MSS_BEGIN(bool);

            MSS(!!dataset_);
            auto &dataset = *dataset_;

            ImGui::Text("Nr records: %d", dataset.records.size());

            {
                auto &wnd = io_.goc();
                Transform t(wnd, 3,1);
                for (const auto &r: dataset.records)
                {
                    if (false) {}
                    else if (r.has_dim(0,1) && r.has_dim(1,1))
                    {
                        const auto &x = r.data(0);
                        const auto &y = r.data(1);
                        io_.dot(3, sf::Color::Green, t(x,y));
                    }
                    else if (r.has_dim(0,2) && r.has_dim(1,2))
                    {
                        io_.arrow(sf::Color::Green, t(r.data(0,0), r.data(0,1)), sf::Vector2f(r.data(1,0), r.data(1,1))*20.0f);
                    }
                }
            }
            MSS_END();
        }
        bool display_cost_()
        {
            MSS_BEGIN(bool);

            MSS(!!model_);
            auto &model = *model_;

            MSS(!!dataset_);
            auto &dataset = *dataset_;

            auto &learn = goc_learn_();

            double data_ll = 0.0;
            {
                auto &simulator = goc_simulator_();
                for (const auto &r: dataset.records)
                {
                    const auto &x = r.fields[0];
                    const auto &y = r.fields[1];
                    std::copy(RANGE(x), model.states.begin()+model.input);
                    std::copy(RANGE(y), model.states.begin()+model.wanted_output);
                    simulator.forward(model.states.data(), model.weights.data());
                    data_ll += model.states[model.loglikelihood];
                }
                data_ll /= dataset.records.size();
                ImGui::Text("data cost: %f", (float)-data_ll);
            }

            double weights_ll = 0.0;
            unsigned int nr_weights = 0;
            {
                for (auto lix = 0u; lix < model.structure.layers.size(); ++lix)
                {
                    for (auto nix = 0u; nix < model.structure.layers[lix].neurons.size(); ++nix)
                    {
                        {
                            const auto stddev = model.structure.neuron(lix, nix).weight_stddev;
                            const auto &weights = model.parameters.neuron(lix, nix).weights;
                            for (const auto weight: weights)
                            {
                                weights_ll += -0.5*(weight*weight)/(stddev*stddev);
                                learn.output << weight << ' ';
                            }
                            nr_weights += weights.size();
                        }
                        {
                            const auto stddev = model.structure.neuron(lix, nix).bias_stddev;
                            const auto bias = model.parameters.neuron(lix, nix).bias;
                            weights_ll += -0.5*(bias*bias)/(stddev*stddev);
                            learn.output << bias << ' ';
                            ++nr_weights;
                        }
                    }
                }
                weights_ll /= nr_weights;
                ImGui::Text("weight cost: %f", (float)-weights_ll);
            }
            learn.total_ll = data_ll + weights_ll;
            ImGui::Text("total cost: %f", (float)-learn.total_ll);
            learn.output << -data_ll << ' ' << -weights_ll << ' ' << -learn.total_ll << std::endl;

            {
                auto &costs = learn.costs;
                for (auto ix = 1u; ix < costs.size(); ++ix)
                    costs[ix-1] = costs[ix];
                costs.back() = -data_ll-weights_ll;
                ImGui::PlotLines("cost", costs.data(), costs.size(), 0, nullptr, 0.0, FLT_MAX, ImVec2(0,100));
                const auto p = std::minmax_element(RANGE(costs));
                ImGui::Text("min: %f, max: %f", *p.first, *p.second);
                ImGui::SameLine();
                if (ImGui::Button("reset cost"))
                    std::fill(RANGE(costs), costs.back());
            }
            ImGui::Separator();

            MSS_END();
        }
        bool learn_model_()
        {
            MSS_BEGIN(bool);

            MSS(!!model_);
            auto &model = *model_;

            MSS(!!dataset_);
            auto &dataset = *dataset_;

            auto &learn = goc_learn_();

            if (ImGui::Checkbox("Learn", &learn.do_learn))
            {
                if (learn.do_learn)
                {
                    //We just start learning: open the output stream
                    if (!learn.output_fn.empty())
                        learn.output.open(learn.output_fn);
                }
                else
                    learn.output_fn.clear();
            }

            //Specifying the output filename is only allowed when learning is pauzed
            if (!learn.do_learn)
            {
                learn.output.close();

                auto &output_fn = learn.output_fn;
                const auto bufsize = 1024;
                char buffer[bufsize];
                const auto size = std::min<unsigned int>(output_fn.size(), bufsize-1);
                output_fn.copy(buffer, size);
                buffer[size] = '\0';
                ImGui::SameLine();
                if (ImGui::InputText("Output filename", buffer, bufsize))
                    output_fn = buffer;
            }

            auto &simulator = goc_simulator_();

            //Create the trainer, if needed
            if (!learn.trainer)
            {
                learn.trainer.emplace(model.structure.nr_inputs, model.structure.nr_outputs());
                auto &trainer = *learn.trainer;
                for (const auto &r: dataset.records)
                {
                    MSS(trainer.add(r.fields[0], r.fields[1]));
                }
                MSS(trainer.set(&simulator, model.input, model.output));
                trainer.add_fixed_input(model.bias, 1.0);
            }

            {
                auto &trainer = *learn.trainer;

                if (ImGui::RadioButton("NoLearn", learn.algo == Algo::NoLearn))
                    learn.algo = Algo::NoLearn;
                ImGui::SameLine();
                if (ImGui::RadioButton("Exhaustive", learn.algo == Algo::Exhaustive))
                {
                    learn.algo = Algo::Exhaustive;
                    learn.current_ixs.clear();
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("Metropolis", learn.algo == Algo::Metropolis))
                    learn.algo = Algo::Metropolis;

                if (ImGui::RadioButton("Steepest descent", learn.algo == Algo::SteepestDescent))
                    learn.algo = Algo::SteepestDescent;
                ImGui::SameLine();
                if (ImGui::RadioButton("Scaled Conjugate Gradient", learn.algo == Algo::SCG))
                    learn.algo = Algo::SCG;
                ImGui::SameLine();
                if (ImGui::RadioButton("ADAM", learn.algo == Algo::Adam))
                    learn.algo = Algo::Adam;

                switch (learn.algo)
                {
                    case Algo::NoLearn:
                        break;
                    case Algo::Exhaustive:
                        {
                            unsigned int nr_dim = 0;
                            for (const auto &p: learn.exhaustive_map)
                                if (p.second)
                                    ++nr_dim;
                            ImGui::Text("Nr dimensions to learn exhaustively: %d", nr_dim);

                            if (ImGui::SliderInt("Exhaustive points to scan", &learn.exhaustive_nr, 2, 100))
                                learn.current_ixs.clear();
                            learn.exhaustive_nr = std::max(learn.exhaustive_nr, 2);

                            if (learn.do_learn)
                            {
                                const auto d = (3.0 - -3.0)/(learn.exhaustive_nr-1);
                                auto trans = [&](unsigned int ix){
                                    return -3.0 + d*ix; 
                                };

                                auto set_weights = [&](const auto &ixs){
                                    auto ix_it = ixs.begin();
                                    auto weight_it = model.weights.begin();
                                    auto assign_if_selected = [&](const auto &neuron, unsigned int lix, unsigned int nix){
                                        for (auto wix = 0u; wix < neuron.weights.size()+1; ++wix)
                                        {
                                            const auto ix = *ix_it++;
                                            if (ix >= 0)
                                                *weight_it = trans(ix);
                                            ++weight_it;
                                        }
                                    };
                                    model.parameters.each_neuron(assign_if_selected);
                                };

                                if (learn.current_ixs.empty())
                                {
                                    //Setup the current_ixs based on the selected weights/biases to use during exhaustive learning
                                    learn.current_ixs.resize(simulator.nr_weights());
                                    auto ix = learn.current_ixs.begin();
                                    auto enable_if_selected = [&](const auto &neuron, unsigned int lix, unsigned int nix){
                                        for (auto wix = 0u; wix < neuron.weights.size(); ++wix)
                                            *ix++ = learn.exhaustive_map[ParamIX(lix, nix, wix)] ? 0 : -1;
                                        *ix++ = learn.exhaustive_map[ParamIX(lix, nix, -1)] ? 0 : -1;
                                    };
                                    model.parameters.each_neuron(enable_if_selected);

                                    set_weights(learn.current_ixs);

                                    learn.best_ixs.clear();
                                }
                                else
                                {
                                    for (auto ix: learn.current_ixs)
                                        std::cout << ix << ' ';
                                    std::cout << std::endl;

                                    if (learn.best_ixs.empty() || learn.total_ll > learn.best_ll)
                                    {
                                        learn.best_ixs = learn.current_ixs;
                                        learn.best_ll = learn.total_ll;
                                    }

                                    //Increment current_ixs
                                    bool carry = true;
                                    for (auto &ix: learn.current_ixs)
                                    {
                                        if (ix == -1)
                                            //Skip this ix
                                            continue;
                                        if (carry)
                                            ++ix;
                                        carry = (ix >= learn.exhaustive_nr);
                                        if (carry)
                                            ix = 0;
                                        else
                                            break;
                                    }

                                    set_weights(learn.current_ixs);

                                    if (carry)
                                    {
                                        learn.do_learn = false;
                                        set_weights(learn.best_ixs);
                                        learn.current_ixs.clear();
                                    }
                                }
                            }
                        }
                        break;
                    case Algo::SteepestDescent:
                        {
                            ImGui::SliderFloat("Steepest descent step size", &learn.sd_step, 0.0, 0.01);

                            ImGui::SliderFloat("Steepest descent max norm", &learn.sd_max_norm, 0.0, 100.0);
                            trainer.set_max_gradient_norm(learn.sd_max_norm);

                            if (learn.do_learn)
                            {
                                double newlp;
                                MSS(trainer.train_sd(newlp, model.weights.data(), model.cost_stddev, model.weights_stddev, learn.sd_step));
                            }
                        }
                        break;
                    case Algo::SCG:
                        {
                            if (ImGui::Button("Reinit SCG"))
                                learn.scg_do_init;
                            if (learn.scg_do_init)
                                trainer.init_scg();
                            learn.scg_do_init = false;

                            if (learn.do_learn)
                            {
                                double newlp;
                                MSS(trainer.train_scg(newlp, model.weights.data(), model.cost_stddev, model.weights_stddev, 10));
                            }
                        }
                        break;
                    case Algo::Adam:
                        {
                            if (ImGui::SliderFloat("Adam step size", &learn.adam_alpha, 0.0001, 0.1))
                                learn.adam_do_init = true;
                            if (ImGui::SliderFloat("Adam decay", &learn.adam_beta1, 0.8, 1.0))
                                learn.adam_do_init = true;

                            if (learn.adam_do_init)
                            {
                                gubg::neural::Trainer<double>::AdamParams adam;
                                adam.alpha = learn.adam_alpha;
                                adam.beta1 = learn.adam_beta1;
                                trainer.init_adam(adam);
                                learn.adam_do_init = false;
                            }

                            if (learn.do_learn)
                            {
                                double newlp;
                                if (!trainer.train_adam(newlp, model.weights.data(), model.cost_stddev, model.weights_stddev))
                                    learn.adam_do_init = true;
                            }
                        }
                        break;
                    case Algo::Metropolis:
                        {
                            learn.metropolis_stddev = std::max(learn.metropolis_stddev, 0.0001f);
                            ImGui::SliderFloat("Metropolis motion stddev", &learn.metropolis_stddev, 0.0001f, 0.1f);

                            if (learn.do_learn)
                            {
                                double newlp;
                                MSS(trainer.train_metropolis(newlp, model.weights.data(), model.cost_stddev, model.weights_stddev, learn.metropolis_stddev, 100));
                            }
                        }
                        break;
                }
                MSS(gubg::neural::copy_weights(model.parameters, model.weights));
            }

            MSS_END();
        }
        void load_font_()
        {
            font_.emplace();
            std::string fn = "GenBasR.ttf";
            {
                auto gubg = std::getenv("gubg");
                if (!!gubg)
                {
                    fn = gubg;
                    fn += "/fonts/GenBasR.ttf";
                }
            }
            if (!font_->loadFromFile(fn))
            {
                error_ << "Could not load the font from " << fn;
                font_.reset();
            }
        }

        //Help functionality to create C-string with proper lifetime management
        gubg::imgui::CStringMgr cstr_;

        std::ostringstream error_;

        std::filesystem::path structure_fn_;
        struct Model
        {
            gubg::mlp::Structure structure;
            gubg::mlp::Parameters parameters;
            unsigned int lix = 0;
            unsigned int nix = 0;
            std::optional<gubg::neural::Simulator<double>> simulator;
            size_t input, bias, output;
            float weight_scale = 1.0f;
            std::vector<double> weights, states;
            double weights_stddev = 3.0;
            double cost_stddev = 0.1;
            double randomize_weights_stddev = 1.0;
            size_t wanted_output, loglikelihood;
        };
        std::optional<Model> model_;

        gubg::neural::Simulator<double> &goc_simulator_()
        {
            if (!model_->simulator)
            {
                auto &model = *model_;
                model.simulator.emplace();
                auto &simulator = *model.simulator;

                //Create the simulator, weights and states
                {
                    gubg::neural::setup(simulator, model.structure, model.input, model.bias, model.output);
                    const auto nr_outputs = model.structure.nr_outputs();
                    model.wanted_output = simulator.add_external(nr_outputs);
                    std::vector<size_t> wanted_outputs(nr_outputs); std::iota(RANGE(wanted_outputs), model.wanted_output);
                    std::vector<size_t> actual_outputs(nr_outputs); std::iota(RANGE(actual_outputs), model.output);
                    simulator.add_loglikelihood(wanted_outputs, actual_outputs, model.loglikelihood, model.cost_stddev);
                    model.weights.resize(simulator.nr_weights());
                    model.states.resize(simulator.nr_states());
                    model.states[model.bias] = 1.0;
                }
            }
            return *model_->simulator;
        }

        std::filesystem::path data_fn_;
        std::optional<DataSet> dataset_;

        struct ParamIX
        {
            unsigned int lix;
            unsigned int nix;
            int wix;
            ParamIX(unsigned int lix, unsigned int nix, int wix): lix(lix), nix(nix), wix(wix) {}
            bool operator<(const ParamIX &rhs) const
            {
                if (lix < rhs.lix)
                    return true;
                if (lix == rhs.lix)
                {
                    if (nix < rhs.nix)
                        return true;
                    if (nix == rhs.nix)
                    {
                        if (wix < rhs.wix)
                            return true;
                    }
                }
                return false;
            }
        };

        enum class Algo {NoLearn, Exhaustive, Metropolis, SteepestDescent, SCG, Adam};
        struct Learn
        {
            bool do_learn = false;
            std::string output_fn;
            std::ofstream output;
            double total_ll = 0.0;;
            std::array<float, 1000> costs{};
            std::optional<gubg::neural::Trainer<double>> trainer;
            Algo algo = Algo::NoLearn;

            float sd_step = 0.003;
            float sd_max_norm = 100.0;

            bool scg_do_init = true;

            bool adam_do_init = true;
            float adam_alpha = 0.01;
            float adam_beta1 = 0.9;

            float metropolis_stddev = 0.001;

            std::map<ParamIX, bool> exhaustive_map;
            int exhaustive_nr = 10;
            std::vector<int> current_ixs;
            double best_ll;
            std::vector<int> best_ixs;
        };
        std::optional<Learn> learn_;
        Learn &goc_learn_()
        {
            if (!learn_)
                learn_.emplace();
            return *learn_;
        }

        struct Pane
        {
            bool valid = false;
            sf::Color color;
            std::string caption;
            double xpos, ypos;
            sf::Text text;
            sf::RenderTexture rt;
            sf::Sprite sprite;
            std::list<Line> lines;
            std::list<Dot> dots;
            std::list<Arrow> arrows;
            void setup(const std::string &caption, unsigned int width, unsigned int height, double xpos, sf::Color color, const std::optional<sf::Font> &font)
            {
                this->caption = caption;
                this->color = color;
                this->xpos = xpos;
                this->ypos = height;
                if (font)
                {
                    text.setFont(*font);
                    text.setString(caption);
                    text.setCharacterSize(24);
                }
                rt.create(width, height);
            }
            sf::RenderTexture &goc()
            {
                if (!valid)
                {
                    valid = true;
                    rt.clear(color);
                    rt.draw(text);
                    lines.clear();
                    dots.clear();
                    arrows.clear();
                }
                return rt;
            }
            template <typename Ftor>
            void line(double width, const sf::Color &color, Ftor &&ftor)
            {
                lines.emplace_back(width, color);
                ftor(lines.back());
            }
            void dot(double width, const sf::Color &color, const sf::Vector2f &pos)
            {
                dots.emplace_back(width, color, pos);
            }
            void arrow(const sf::Color &color, const sf::Vector2f &pos, const sf::Vector2f &dir)
            {
                arrows.emplace_back(color, pos, dir);
            }
            void draw(sf::RenderWindow &wnd)
            {
                for (const auto &line: lines)
                    line.draw(rt);
                for (const auto &dot: dots)
                    dot.draw(rt);
                for (const auto &arrow: arrows)
                    arrow.draw(rt);
                sprite.setPosition(sf::Vector2f(xpos, ypos));
                sprite.setScale(1.0, -1.0);
                sprite.setTexture(rt.getTexture());
                wnd.draw(sprite);
                valid = false;
            }
        };
        Pane io_;

        std::optional<sf::Font> font_;

        std::mt19937 rng_;
    };

} 

#endif
