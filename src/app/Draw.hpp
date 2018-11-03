#ifndef HEADER_app_Draw_hpp_ALREADY_INCLUDED
#define HEADER_app_Draw_hpp_ALREADY_INCLUDED

#include "gubg/std/optional.hpp"
#include "gubg/debug.hpp"
#include "SFML/Graphics.hpp"
#include <vector>
#include <array>
#include <cmath>

namespace app { 

    class Line
    {
    public:
        Line(double width, sf::Color color): width_(width), color_(color) {}

        Line &point(const sf::Vector2f &pos)
        {
            auto cur = pos;
            if (prev_)
            {
                const auto &prev = *prev_;
                auto diff = cur - prev;
                auto norm = sf::Vector2f(-diff.y, diff.x);
                norm *= (float)width_/std::sqrt(norm.x*norm.x + norm.y*norm.y);
                vertices_.push_back(sf::Vertex(prev+norm, color_));
                vertices_.push_back(sf::Vertex(prev-norm, color_));
                vertices_.push_back(sf::Vertex(cur+norm, color_));
                vertices_.push_back(sf::Vertex(cur-norm, color_));
            }
            prev_ = cur;
            return *this;
        }

        template <typename Window>
        void draw(Window &wnd) const
        {
            wnd.draw(vertices_.data(), vertices_.size(), sf::TrianglesStrip);
        }

    private:
        double width_;
        sf::Color color_;
        std::optional<sf::Vector2f> prev_;
        std::vector<sf::Vertex> vertices_;
    };

    class Dot
    {
    public:
        Dot(double width, sf::Color color, sf::Vector2f pos)
        {
            pos.x += width; pos.y += width;     vertices_[0] = sf::Vertex(pos, color);
            pos.x -= 2*width;                   vertices_[1] = sf::Vertex(pos, color);
            pos.x += 2*width; pos.y -= 2*width; vertices_[2] = sf::Vertex(pos, color);
            pos.x -= 2*width;                   vertices_[3] = sf::Vertex(pos, color);
        }

        template <typename Window>
        void draw(Window &wnd) const
        {
            wnd.draw(vertices_.data(), vertices_.size(), sf::TrianglesStrip);
        }
    private:
        std::array<sf::Vertex, 4> vertices_;
    };

    class Arrow
    {
    public:
        Arrow(sf::Color color, sf::Vector2f pos, sf::Vector2f dir)
        {
            S("");L(C(pos.x)C(pos.y)C(dir.x)C(dir.y));
            auto norm = sf::Vector2f(-dir.y, dir.x);
            vertices_[0] = sf::Vector2f(pos+norm*0.2f);
            vertices_[1] = sf::Vector2f(pos-norm*0.2f);
            vertices_[2] = sf::Vector2f(pos+dir);
            for (auto &v: vertices_)
                v.color = color;
        }

        template <typename Window>
        void draw(Window &wnd) const
        {
            wnd.draw(vertices_.data(), vertices_.size(), sf::Triangles);
        }
    private:
        std::array<sf::Vertex, 3> vertices_;
    };
} 

#endif
