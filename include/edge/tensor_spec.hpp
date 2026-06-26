#pragma once

#include <cstddef>

namespace edge {

enum class Layout {
    Flat,
    CHW,
    HWC,
    TimeFeature,
    TokenFeature,
    HeadTokenFeature
};

template<std::size_t... Dims>
struct Shape {
    static_assert(sizeof...(Dims) > 0, "Shape must have at least one dimension");
    static_assert(((Dims > 0U) && ...), "Shape dimensions must be greater than zero");

    static constexpr std::size_t rank = sizeof...(Dims);
    static constexpr std::size_t elements = (Dims * ... * 1U);
};

namespace detail {

template<std::size_t Index, std::size_t First, std::size_t... Rest>
struct ShapeDimAt {
    static constexpr std::size_t value = ShapeDimAt<Index - 1U, Rest...>::value;
};

template<std::size_t First, std::size_t... Rest>
struct ShapeDimAt<0U, First, Rest...> {
    static constexpr std::size_t value = First;
};

template<std::size_t Index, typename ShapeT>
struct ShapeDim;

template<std::size_t Index, std::size_t... Dims>
struct ShapeDim<Index, Shape<Dims...>> {
    static_assert(Index < sizeof...(Dims), "Shape dimension index is out of range");
    static constexpr std::size_t value = ShapeDimAt<Index, Dims...>::value;
};

} // namespace detail

template<std::size_t Index, typename ShapeT>
inline constexpr std::size_t shape_dim_v = detail::ShapeDim<Index, ShapeT>::value;

template<typename ShapeT, Layout LayoutV>
struct TensorSpec {
    using shape = ShapeT;
    static constexpr Layout layout = LayoutV;
    static constexpr std::size_t rank = ShapeT::rank;
    static constexpr std::size_t elements = ShapeT::elements;
};

template<std::size_t Features>
using Vector = TensorSpec<Shape<Features>, Layout::Flat>;

template<std::size_t Channels, std::size_t Height, std::size_t Width>
using CHW = TensorSpec<Shape<Channels, Height, Width>, Layout::CHW>;

template<std::size_t Height, std::size_t Width, std::size_t Channels>
using HWC = TensorSpec<Shape<Height, Width, Channels>, Layout::HWC>;

template<typename TensorSpecT>
struct Input {
    using spec = TensorSpecT;
    static constexpr std::size_t features = spec::elements;
};

template<std::size_t Features>
using InputVector = Input<Vector<Features>>;

} // namespace edge
