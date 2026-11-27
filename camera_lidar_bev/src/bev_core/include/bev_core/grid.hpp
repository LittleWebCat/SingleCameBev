#ifndef BEV_CORE__GRID_HPP_
#define BEV_CORE__GRID_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace bev_core
{

// Coordinates use the vehicle convention: x forward, y left, z up.
// Both ranges are half-open: [minimum, maximum).
struct GridSpec
{
  double x_min_m{0.0};
  double x_max_m{40.0};
  double y_min_m{-15.0};
  double y_max_m{15.0};
  double resolution_m{0.20};
};

enum class Occupancy : std::int8_t
{
  Unknown = -1,
  Free = 0,
  Occupied = 100
};

enum class SemanticClass : std::uint8_t
{
  Unknown = 0,
  Road = 1,
  Sidewalk = 2,
  Vehicle = 3,
  Pedestrian = 4,
  Cyclist = 5,
  Other = 6
};

struct Cell
{
  Occupancy occupancy{Occupancy::Unknown};
  SemanticClass semantic{SemanticClass::Unknown};
  double ground_height_m{std::numeric_limits<double>::quiet_NaN()};
  double obstacle_height_m{std::numeric_limits<double>::quiet_NaN()};
  // Seconds in a common sensor time basis, to be supplied by future integration.
  // NaN means unobserved; a publication timestamp is not an observation time.
  double observation_time_seconds{std::numeric_limits<double>::quiet_NaN()};
  float confidence{0.0F};
};

struct CellIndex
{
  std::size_t forward;
  std::size_t lateral;
};

class Grid
{
public:
  // Throws std::invalid_argument for nonfinite, inverted, or nonintegral specs.
  // Allocation failures propagate normally.
  explicit Grid(GridSpec spec = {});

  const GridSpec & spec() const noexcept;
  std::size_t forward_cells() const noexcept;
  std::size_t lateral_cells() const noexcept;
  std::size_t cell_count() const noexcept;

  // Flat storage is forward-major: forward * lateral_cells() + lateral.
  // y (lateral) varies fastest; this is not nav_msgs/OccupancyGrid ordering.
  const std::vector<Cell> & cells() const noexcept;

  // Nonfinite coordinates and points outside the half-open bounds return nullopt.
  std::optional<CellIndex> cell_for_point(double x_m, double y_m) const noexcept;
  std::optional<std::size_t> index_for_point(double x_m, double y_m) const noexcept;

  // Index-based methods throw std::out_of_range for invalid indices.
  std::size_t linear_index(CellIndex index) const;
  CellIndex cell_index(std::size_t index) const;
  std::pair<double, double> cell_center(CellIndex index) const;

  const Cell & cell(CellIndex index) const;
  Cell & cell(CellIndex index);
  const Cell & cell(std::size_t index) const;
  Cell & cell(std::size_t index);

  // Clear all evidence back to unknown with invalid heights and observation time.
  void reset() noexcept;

private:
  GridSpec spec_;
  std::size_t forward_cells_;
  std::size_t lateral_cells_;
  std::vector<Cell> cells_;
};

}  // namespace bev_core

#endif  // BEV_CORE__GRID_HPP_
