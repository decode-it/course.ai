#ifndef HEADER_app_Util_hpp_ALREADY_INCLUDED
#define HEADER_app_Util_hpp_ALREADY_INCLUDED

#include "SFML/Graphics.hpp"

namespace app { 

    class Linear
    {
    public:
        Linear(double from, double to): b_(to/2), a_(b_/from) {}
        double operator()(double v) const {return a_*v+b_;}
    private:
        double b_;
        double a_;
    };

    class Transform
    {
    public:
        template <typename Window>
        Transform(const Window &wnd, double x, double y): x_(x, wnd.getSize().x), y_(-y, wnd.getSize().y) {}
        sf::Vector2f operator()(double x, double y) const {return sf::Vector2f(x_(x), y_(y));}
    private:
        Linear x_, y_;
    };

} 

#endif
