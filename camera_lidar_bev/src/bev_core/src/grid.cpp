#include "bev_core/grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace bev_core
{
namespace
{

std::size_t dimension(double minimum, double maximum, double resolution, const char * axis)
{
  if (!std::isfinite(minimum) || !std::isfinite(maximum) || maximum <= minimum) {
    throw std::invalid_argument(std::string(axis) + " bounds must be finite and increasing");
  }
  const double count = (maximum - minimum) / resolution;
  const double rounded = std::round(count);
  if (!std::isfinite(count) || rounded < 1.0 ||
    rounded >= static_cast<double>(std::numeric_limits<std::size_t>::max()))
  {
    throw std::invalid_argument(std::string(axis) + " cell count is out of range");
  }
  // Permit floating-point roundoff in a mathematically integral cell count.
  const double tolerance = 32.0 * std::numeric_limits<double>::epsilon() *
    std::max(1.0, std::abs(count));
  if (std::abs(count - rounded) > tolerance) {
    throw std::invalid_argument(std::string(axis) + " extent must be divisible by resolution");
  }
  return static_cast<std::size_t>(rounded);
}

}  // namespace

Grid::Grid(GridSpec spec)
: spec_(spec), forward_cells_(0), lateral_cells_(0)
{
  if (!std::isfinite(spec_.resolution_m) || spec_.resolution_m <= 0.0) {
    throw std::invalid_argument("resolution must be finite and positive");
  }
  forward_cells_ = dimension(spec_.x_min_m, spec_.x_max_m, spec_.resolution_m, "x");
  lateral_cells_ = dimension(spec_.y_min_m, spec_.y_max_m, spec_.resolution_m, "y");
  if (forward_cells_ > cells_.max_size() / lateral_cells_) {
    throw std::invalid_argument("grid cell count exceeds storage capacity");
  }
  cells_.resize(forward_cells_ * lateral_cells_);
}

const GridSpec & Grid::spec() const noexcept { return spec_; }
std::size_t Grid::forward_cells() const noexcept { return forward_cells_; }
std::size_t Grid::lateral_cells() const noexcept { return lateral_cells_; }
std::size_t Grid::cell_count() const noexcept { return cells_.size(); }
const std::vector<Cell> & Grid::cells() const noexcept { return cells_; }

std::optional<CellIndex> Grid::cell_for_point(double x_m, double y_m) const noexcept
{
  if (!std::isfinite(x_m) || !std::isfinite(y_m) ||
    x_m < spec_.x_min_m || x_m >= spec_.x_max_m ||
    y_m < spec_.y_min_m || y_m >= spec_.y_max_m)
  {
    return std::nullopt;
  }
  // Cancellation can round a point immediately below the maximum to count.
  // The explicit bounds check above lets us clamp that final floating-point cell.
  const double forward = std::floor((x_m - spec_.x_min_m) / spec_.resolution_m);
  const double lateral = std::floor((y_m - spec_.y_min_m) / spec_.resolution_m);
  return CellIndex{
    static_cast<std::size_t>(std::min(forward, static_cast<double>(forward_cells_ - 1))),
    static_cast<std::size_t>(std::min(lateral, static_cast<double>(lateral_cells_ - 1)))};
}

std::optional<std::size_t> Grid::index_for_point(double x_m, double y_m) const noexcept
{
  const auto index = cell_for_point(x_m, y_m);
  if (!index) {
    return std::nullopt;
  }
  return index->forward * lateral_cells_ + index->lateral;
}

std::size_t Grid::linear_index(CellIndex index) const
{
  if (index.forward >= forward_cells_ || index.lateral >= lateral_cells_) {
    throw std::out_of_range("grid cell index is out of range");
  }
  return index.forward * lateral_cells_ + index.lateral;
}

CellIndex Grid::cell_index(std::size_t index) const
{
  if (index >= cells_.size()) {
    throw std::out_of_range("grid linear index is out of range");
  }
  return CellIndex{index / lateral_cells_, index % lateral_cells_};
}

std::pair<double, double> Grid::cell_center(CellIndex index) const
{
  (void)linear_index(index);
  return {
    spec_.x_min_m + (static_cast<double>(index.forward) + 0.5) * spec_.resolution_m,
    spec_.y_min_m + (static_cast<double>(index.lateral) + 0.5) * spec_.resolution_m};
}

const Cell & Grid::cell(CellIndex index) const { return cells_[linear_index(index)]; }
Cell & Grid::cell(CellIndex index) { return cells_[linear_index(index)]; }
const Cell & Grid::cell(std::size_t index) const { return cells_.at(index); }
Cell & Grid::cell(std::size_t index) { return cells_.at(index); }

void Grid::reset() noexcept { std::fill(cells_.begin(), cells_.end(), Cell{}); }

}  // namespace bev_core
