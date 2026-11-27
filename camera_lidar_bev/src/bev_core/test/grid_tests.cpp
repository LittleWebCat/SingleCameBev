#include "bev_core/grid.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

void require(bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template<typename Exception, typename Function>
void require_throws(Function function, const std::string & message)
{
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

void require_unknown(const bev_core::Cell & cell)
{
  require(cell.occupancy == bev_core::Occupancy::Unknown, "unobserved occupancy must be unknown");
  require(cell.semantic == bev_core::SemanticClass::Unknown, "unobserved semantics must be unknown");
  require(std::isnan(cell.ground_height_m), "unobserved ground height must be invalid");
  require(std::isnan(cell.obstacle_height_m), "unobserved obstacle height must be invalid");
  require(std::isnan(cell.observation_time_seconds), "unobserved timestamp must be invalid");
  require(cell.confidence == 0.0F, "unobserved confidence must be zero");
}

void geometry_and_defaults()
{
  bev_core::Grid grid;
  require(grid.forward_cells() == 200, "default forward dimension must be 200");
  require(grid.lateral_cells() == 150, "default lateral dimension must be 150");
  require(grid.cell_count() == 30000, "default grid must have 30000 cells");
  require(static_cast<int>(bev_core::Occupancy::Unknown) == -1, "unknown occupancy encoding");
  require(static_cast<int>(bev_core::Occupancy::Free) == 0, "free occupancy encoding");
  require(static_cast<int>(bev_core::Occupancy::Occupied) == 100, "occupied occupancy encoding");
  for (const auto & cell : grid.cells()) {
    require_unknown(cell);
  }

  require(grid.linear_index({1, 2}) == 152, "lateral coordinate must vary fastest");
  require(grid.linear_index({199, 149}) == 29999, "last row-major index");
  for (std::size_t index = 0; index < grid.cell_count(); ++index) {
    const auto coordinates = grid.cell_index(index);
    require(grid.linear_index(coordinates) == index, "all linear indices must round-trip");
    const auto center = grid.cell_center(coordinates);
    const auto recovered = grid.index_for_point(center.first, center.second);
    require(recovered && *recovered == index, "all cell centers must map back to their cells");
  }
  const auto first_center = grid.cell_center({0, 0});
  require(std::abs(first_center.first - 0.1) < 1e-12, "first x center");
  require(std::abs(first_center.second + 14.9) < 1e-12, "first y center");

  auto & cell = grid.cell({2, 3});
  cell.occupancy = bev_core::Occupancy::Occupied;
  cell.semantic = bev_core::SemanticClass::Vehicle;
  cell.ground_height_m = 1.0;
  cell.obstacle_height_m = 2.0;
  cell.observation_time_seconds = 12.5;
  cell.confidence = 0.8F;
  require(grid.cell(303).semantic == bev_core::SemanticClass::Vehicle, "cell access must share storage");
  grid.reset();
  for (const auto & reset_cell : grid.cells()) {
    require_unknown(reset_cell);
  }
}

void boundaries_and_nonfinite_points()
{
  bev_core::Grid grid;
  const auto minimum = grid.cell_for_point(0.0, -15.0);
  require(minimum && minimum->forward == 0 && minimum->lateral == 0, "minimum bounds are included");
  const auto maximum_inside = grid.cell_for_point(
    std::nextafter(40.0, 0.0), std::nextafter(15.0, 0.0));
  require(maximum_inside && maximum_inside->forward == 199 && maximum_inside->lateral == 149,
    "points immediately below maximum bounds belong to the last cell");
  require(!grid.cell_for_point(40.0, 0.0), "maximum forward bound is excluded");
  require(!grid.cell_for_point(1.0, 15.0), "maximum lateral bound is excluded");
  require(!grid.cell_for_point(std::nextafter(0.0, -1.0), 0.0), "point below x minimum is excluded");
  require(!grid.cell_for_point(1.0, std::nextafter(-15.0, -16.0)), "point below y minimum is excluded");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double invalid : {nan, inf, -inf}) {
    require(!grid.cell_for_point(invalid, 0.0), "nonfinite x must be rejected");
    require(!grid.cell_for_point(1.0, invalid), "nonfinite y must be rejected");
    require(!grid.index_for_point(invalid, 0.0), "invalid point must have no linear index");
  }

  // An exactly representable resolution verifies interior half-open boundaries.
  bev_core::Grid shifted({-1.0, 1.0, -2.0, 2.0, 0.25});
  const auto boundary = shifted.cell_for_point(-0.75, -1.75);
  require(boundary && boundary->forward == 1 && boundary->lateral == 1,
    "interior boundary belongs to the next cell");
  const auto before = shifted.cell_for_point(
    std::nextafter(-0.75, -1.0), std::nextafter(-1.75, -2.0));
  require(before && before->forward == 0 && before->lateral == 0,
    "point immediately before interior boundary belongs to previous cell");
}

void invalid_configurations_and_indices()
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double invalid : {0.0, -0.2, nan, inf}) {
    bev_core::GridSpec spec;
    spec.resolution_m = invalid;
    require_throws<std::invalid_argument>([&] { bev_core::Grid grid(spec); }, "invalid resolution must throw");
  }
  for (const auto & spec : {
      bev_core::GridSpec{0.0, 0.0, -15.0, 15.0, 0.2},
      bev_core::GridSpec{40.0, 0.0, -15.0, 15.0, 0.2},
      bev_core::GridSpec{0.0, 40.0, 15.0, -15.0, 0.2},
      bev_core::GridSpec{nan, 40.0, -15.0, 15.0, 0.2},
      bev_core::GridSpec{0.0, inf, -15.0, 15.0, 0.2},
      bev_core::GridSpec{0.0, 40.0, -15.0, nan, 0.2},
      bev_core::GridSpec{0.0, 40.1, -15.0, 15.0, 0.2},
      bev_core::GridSpec{0.0, 40.0, -15.0, 15.1, 0.2},
      bev_core::GridSpec{0.0, 0.1, -15.0, 15.0, 0.2},
      bev_core::GridSpec{-1e308, 1e308, -15.0, 15.0, 0.2},
      bev_core::GridSpec{0.0, 1e20, 0.0, 1.0, 1.0},
      bev_core::GridSpec{0.0, 1e10, 0.0, 1e10, 1.0}})
  {
    require_throws<std::invalid_argument>([&] { bev_core::Grid grid(spec); }, "invalid grid spec must throw");
  }
  bev_core::Grid grid;
  require_throws<std::out_of_range>([&] { grid.linear_index({200, 0}); }, "invalid forward index");
  require_throws<std::out_of_range>([&] { grid.linear_index({0, 150}); }, "invalid lateral index");
  require_throws<std::out_of_range>([&] { grid.cell_index(30000); }, "invalid linear index");
  require_throws<std::out_of_range>([&] { grid.cell_center({200, 0}); }, "invalid cell center index");
  require_throws<std::out_of_range>([&] { grid.cell(30000); }, "invalid cell access");
}

}  // namespace

int main()
{
  try {
    geometry_and_defaults();
    boundaries_and_nonfinite_points();
    invalid_configurations_and_indices();
  } catch (const std::exception & error) {
    std::cerr << "bev_core test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "bev_core geometry, validity, and unknown-state tests passed\n";
  return EXIT_SUCCESS;
}
