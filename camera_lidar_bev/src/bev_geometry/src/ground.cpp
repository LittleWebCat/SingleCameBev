#include "bev_geometry/ground.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace bev_geometry {
namespace {
constexpr double kPi = 3.14159265358979323846;
using Key = std::pair<int, int>;
struct Patch { std::vector<std::size_t> indices; };
struct LocalFloor { double residual; std::size_t count; };
using Patches = std::map<Key, Patch>;
std::uint64_t packed_key(const Key &key) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.first))<<32) |
         static_cast<std::uint32_t>(key.second);
}

bool finite(const Point &p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
Key patch_key(const Point &p, double size) {
  return {static_cast<int>(std::floor(p.x / size)), static_cast<int>(std::floor(p.y / size))};
}
double residual(const Point &p, const GroundModel &g) { return p.z - g.a*p.x - g.b*p.y - g.c; }
double slope(const GroundModel &g) { return std::atan(std::hypot(g.a,g.b)) * 180.0/kPi; }
bool acceptable_height(const GroundModel &g, const Config &c) {
  if (std::isfinite(c.ground_height_m)) {
    return std::abs(g.c + c.ground_height_m) <= c.ground_height_tolerance_m;
  }
  return -g.c >= c.auto_ground_height_min_m && -g.c <= c.auto_ground_height_max_m;
}

// Small least-squares fit of z = ax + by + c, with pivoting and degeneracy checks.
bool fit_plane(const std::vector<Point> &points, GroundModel &out) {
  if (points.size() < 3) { return false; }
  double matrix[3][4]{};
  for (const auto &p : points) {
    const double v[3]{p.x,p.y,1.0};
    for (int row=0;row<3;++row) {
      for (int col=0;col<3;++col) { matrix[row][col] += v[row]*v[col]; }
      matrix[row][3] += v[row]*p.z;
    }
  }
  for (int col=0;col<3;++col) {
    int pivot=col;
    for (int row=col+1;row<3;++row) {
      if (std::abs(matrix[row][col]) > std::abs(matrix[pivot][col])) { pivot=row; }
    }
    if (std::abs(matrix[pivot][col]) < 1e-8) { return false; }
    for (int j=col;j<4;++j) { std::swap(matrix[pivot][j],matrix[col][j]); }
    const double divisor=matrix[col][col];
    for (int j=col;j<4;++j) { matrix[col][j] /= divisor; }
    for (int row=0;row<3;++row) {
      if (row==col) { continue; }
      const double factor=matrix[row][col];
      for (int j=col;j<4;++j) { matrix[row][j] -= factor*matrix[col][j]; }
    }
  }
  out.a=matrix[0][3]; out.b=matrix[1][3]; out.c=matrix[2][3];
  return std::isfinite(out.a) && std::isfinite(out.b) && std::isfinite(out.c);
}

bool broad_support(const std::vector<Point> &points, double min_span) {
  if (points.size()<3) { return false; }
  double xmin=points.front().x, xmax=xmin, ymin=points.front().y, ymax=ymin;
  double xsum=0, ysum=0, xxsum=0, yysum=0, xysum=0;
  for (const auto &p : points) {
    xmin=std::min(xmin,p.x); xmax=std::max(xmax,p.x);
    ymin=std::min(ymin,p.y); ymax=std::max(ymax,p.y);
    xsum+=p.x; ysum+=p.y; xxsum+=p.x*p.x; yysum+=p.y*p.y; xysum+=p.x*p.y;
  }
  const double n=static_cast<double>(points.size());
  const double xx=xxsum/n-(xsum/n)*(xsum/n);
  const double yy=yysum/n-(ysum/n)*(ysum/n);
  const double xy=xysum/n-(xsum/n)*(ysum/n);
  const double minor=0.5*(xx+yy-std::hypot(xx-yy,2*xy));
  // A long wall/stripe must not masquerade as a floor despite large x/y spans.
  return xmax-xmin >= min_span && ymax-ymin >= min_span && minor >= min_span*min_span/48.0;
}

std::vector<std::vector<Point>> connected_support(const std::vector<Point> &points,const Config &config) {
  std::unordered_map<std::uint64_t,std::size_t> lookup;
  lookup.reserve(points.size()*2);
  std::vector<Key> keys;
  keys.reserve(points.size());
  for (std::size_t i=0;i<points.size();++i) {
    keys.push_back(patch_key(points[i],config.patch_size_m));
    lookup.emplace(packed_key(keys.back()),i);
  }
  std::vector<bool> visited(points.size(),false);
  std::vector<std::vector<Point>> components;
  const int neighbor=static_cast<int>(std::ceil(config.local_support_radius_m/config.patch_size_m));
  for (std::size_t first=0;first<points.size();++first) {
    if (visited[first]) { continue; }
    visited[first]=true;
    std::vector<std::size_t> queue{first};
    std::vector<Point> component;
    for (std::size_t cursor=0;cursor<queue.size();++cursor) {
      const auto index=queue[cursor];
      component.push_back(points[index]);
      for (int dx=-neighbor;dx<=neighbor;++dx) {
        for (int dy=-neighbor;dy<=neighbor;++dy) {
          if (std::hypot(static_cast<double>(dx),static_cast<double>(dy))*config.patch_size_m>
              config.local_support_radius_m) { continue; }
          const auto found=lookup.find(packed_key({keys[index].first+dx,keys[index].second+dy}));
          if (found!=lookup.end() && !visited[found->second]) {
            visited[found->second]=true; queue.push_back(found->second);
          }
        }
      }
    }
    components.push_back(std::move(component));
  }
  return components;
}

GroundModel estimate_floor(const std::vector<Point> &points, const Patches &patches,
                           const Config &config) {
  GroundModel best;
  best.used_height_prior=std::isfinite(config.ground_height_m);
  const double maximum_gradient=std::tan(config.max_slope_deg*kPi/180.0);
  std::vector<Point> seeds;
  for (const auto &entry : patches) {
    std::vector<Point> low;
    for (const auto i : entry.second.indices) {
      const auto &p=points[i];
      const double range=std::hypot(p.x,p.y);
      if (range<config.fit_min_range_m || range>config.fit_max_range_m) { continue; }
      // No ceilings and no extrapolation of a wall above the sensor into a floor.
      if (p.z > -config.auto_ground_height_min_m && !best.used_height_prior) { continue; }
      if (best.used_height_prior && std::abs(p.z+config.ground_height_m) >
          config.ground_height_tolerance_m+range*maximum_gradient) { continue; }
      low.push_back(p);
    }
    if (low.size()<3) { continue; }
    const std::size_t pick=low.size()/10;
    std::nth_element(low.begin(),low.begin()+static_cast<std::ptrdiff_t>(pick),low.end(),
                     [](const Point &a,const Point &b){return a.z<b.z;});
    seeds.push_back(low[pick]);
  }
  if (seeds.size()<config.min_ground_patches) {
    best.reason="insufficient low-surface patches"; return best;
  }
  std::mt19937 random(0x424556U);
  std::uniform_int_distribution<std::size_t> pick(0,seeds.size()-1);
  std::vector<Point> winning;
  double winning_error=std::numeric_limits<double>::infinity();
  const auto consider=[&](const GroundModel &candidate) {
    if (slope(candidate)>config.max_slope_deg || !acceptable_height(candidate,config)) { return; }
    std::vector<Point> support;
    support.reserve(seeds.size());
    double error=0;
    for (const auto &p : seeds) {
      const double d=residual(p,candidate);
      if (std::abs(d)<=config.ground_distance_m) { support.push_back(p); error+=d*d; }
    }
    if (support.size()<config.min_ground_patches || !broad_support(support,config.min_ground_span_m)) {
      return;
    }
    if (support.size()<winning.size()) { return; }
    // Distant wall/roof strips cannot borrow a floor intercept and jointly
    // masquerade as one inclined floor. Each connected observed component
    // must satisfy the same data-fit and coverage gates on its own.
    for (auto &component : connected_support(support,config)) {
      if (component.size()<config.min_ground_patches || component.size()<winning.size() ||
          !broad_support(component,config.min_ground_span_m)) { continue; }
      GroundModel refined;
      if (!fit_plane(component,refined) || slope(refined)>config.max_slope_deg ||
          !acceptable_height(refined,config)) { continue; }
      error=0;
      for (const auto &p : component) { const double d=residual(p,refined); error+=d*d; }
      if (component.size()>winning.size() || (component.size()==winning.size() && error<winning_error)) {
        winning=std::move(component); winning_error=error; best=refined;
      }
    }
  };
  if (std::isfinite(config.ground_height_m)) {
    std::vector<std::size_t> height_band;
    for (std::size_t i=0;i<seeds.size();++i) {
      if (std::abs(seeds[i].z+config.ground_height_m)<=config.ground_height_tolerance_m) {
        height_band.push_back(i);
      }
    }
    // The height prior supplies one plane constraint, so only two real floor
    // seeds are needed. Sparse floors amid many walls are otherwise missed by
    // a small three-point RANSAC budget. Sample the whole admissible slope cone
    // as well as the near-height band, retaining inclined floors outside it.
    for (int trial=0;trial<512;++trial) {
      const bool band=trial<256 && height_band.size()>=3;
      const auto choose=[&]() {
        if (!band) { return pick(random); }
        std::uniform_int_distribution<std::size_t> select(0,height_band.size()-1);
        return height_band[select(random)];
      };
      const auto i=choose(), j=choose();
      if (i==j) { continue; }
      const auto &p=seeds[i], &q=seeds[j];
      const double determinant=p.x*q.y-q.x*p.y;
      if (std::abs(determinant)<0.5) { continue; }
      // This is a proposal only: the final fit freely estimates the intercept
      // and must pass all original point, patch, coverage, and slope gates.
      for (int offset=-4;offset<=4;++offset) {
        GroundModel candidate;
        candidate.c=-config.ground_height_m+offset*config.ground_height_tolerance_m/4.0;
        const double u=p.z-candidate.c, v=q.z-candidate.c;
        candidate.a=(u*q.y-v*p.y)/determinant;
        candidate.b=(p.x*v-q.x*u)/determinant;
        consider(candidate);
      }
    }
  }
  for (int trial=0;trial<192;++trial) {
    const auto i=pick(random), j=pick(random), k=pick(random);
    if (i==j || i==k || j==k) { continue; }
    GroundModel candidate;
    if (fit_plane({seeds[i],seeds[j],seeds[k]},candidate)) { consider(candidate); }
  }
  best.used_height_prior=std::isfinite(config.ground_height_m);
  if (winning.empty()) { best.reason="no low floor with sufficient planar coverage"; return best; }
  if (!fit_plane(winning,best)) { best.reason="degenerate floor support"; return best; }
  // Refine with equal patch weight. Dense vehicle surfaces cannot dominate solely by point count.
  for (int pass=0;pass<2;++pass) {
    std::vector<Point> inliers;
    for (const auto &p : seeds) {
      if (std::abs(residual(p,best))<=config.ground_distance_m) { inliers.push_back(p); }
    }
    std::vector<Point> next;
    GroundModel refined;
    for (auto &component : connected_support(inliers,config)) {
      GroundModel fit;
      if (component.size()>=config.min_ground_patches && component.size()>next.size() &&
          broad_support(component,config.min_ground_span_m) && fit_plane(component,fit) &&
          slope(fit)<=config.max_slope_deg && acceptable_height(fit,config)) {
        next=std::move(component); refined=fit;
      }
    }
    if (next.empty()) { best.reason="unstable floor coverage"; return best; }
    winning=std::move(next); best=refined;
  }
  best.used_height_prior=std::isfinite(config.ground_height_m);
  if (slope(best)>config.max_slope_deg || !acceptable_height(best,config)) {
    best.reason="floor slope or sensor-height gate rejected fit"; return best;
  }
  std::unordered_map<std::uint64_t,bool> winning_patches;
  winning_patches.reserve(winning.size()*2);
  for (const auto &seed : winning) { winning_patches.emplace(packed_key(patch_key(seed,config.patch_size_m)),true); }
  for (const auto &entry : patches) {
    if (winning_patches.find(packed_key(entry.first))==winning_patches.end()) { continue; }
    for (const auto index : entry.second.indices) {
      const auto &p=points[index];
      const double range=std::hypot(p.x,p.y);
      if (range<config.fit_min_range_m || range>config.fit_max_range_m) { continue; }
      const double d=residual(p,best);
      if (std::abs(d)<=config.ground_distance_m) { ++best.support_points; }
    }
  }
  best.support_patches=winning.size();
  best.slope_deg=slope(best);
  double squared=0;
  for (const auto &seed : winning) { const double d=residual(seed,best); squared+=d*d; }
  best.rms_m=std::sqrt(squared/static_cast<double>(winning.size()));
  if (best.support_points<config.min_ground_points) {
    best.reason="insufficient floor return count"; return best;
  }
  best.valid=true;
  best.reason=best.used_height_prior?"supported floor within configured height prior":"provisional supported low surface; sensor height unmeasured";
  return best;
}

bool excluded_by_self_mask(const Point &p,const Config &c) {
  return c.self_mask_enabled && p.x>=c.self_x_min_m && p.x<=c.self_x_max_m &&
    p.y>=c.self_y_min_m && p.y<=c.self_y_max_m && p.z>=c.self_z_min_m && p.z<=c.self_z_max_m;
}

// Exact supercover traversal from sensor origin to measured ground endpoint.
// Cells touching a ray at a corner are included conservatively.
bool clear_ground_line(const Point &p,const bev_core::Grid &grid) {
  const auto &spec=grid.spec();
  double entry=0, exit=1;
  const auto clip=[&](double direction,double minimum,double maximum) {
    if (direction==0) { return minimum<=0 && maximum>0; }
    double lo=minimum/direction, hi=maximum/direction;
    if (lo>hi) { std::swap(lo,hi); }
    entry=std::max(entry,lo); exit=std::min(exit,hi);
    return entry<=exit;
  };
  if (!clip(p.x,spec.x_min_m,spec.x_max_m) || !clip(p.y,spec.y_min_m,spec.y_max_m)) { return false; }
  const double first=std::nextafter(entry,exit);
  const auto initial=grid.cell_for_point(p.x*first,p.y*first);
  if (!initial) { return false; }
  int x=static_cast<int>(initial->forward), y=static_cast<int>(initial->lateral);
  const int sx=p.x>0?1:p.x<0?-1:0, sy=p.y>0?1:p.y<0?-1:0;
  const double infinity=std::numeric_limits<double>::infinity();
  double tx=sx==0?infinity:(spec.x_min_m+(x+(sx>0?1:0))*spec.resolution_m)/p.x;
  double ty=sy==0?infinity:(spec.y_min_m+(y+(sy>0?1:0))*spec.resolution_m)/p.y;
  const double dx=sx==0?infinity:spec.resolution_m/std::abs(p.x);
  const double dy=sy==0?infinity:spec.resolution_m/std::abs(p.y);
  const auto blocked=[&](int xi,int yi) {
    return xi>=0 && yi>=0 && static_cast<std::size_t>(xi)<grid.forward_cells() &&
      static_cast<std::size_t>(yi)<grid.lateral_cells() &&
      grid.cell({static_cast<std::size_t>(xi),static_cast<std::size_t>(yi)}).occupancy==bev_core::Occupancy::Occupied;
  };
  const auto limit=grid.forward_cells()+grid.lateral_cells()+4;
  for (std::size_t iteration=0;iteration<limit;++iteration) {
    if (blocked(x,y)) { return false; }
    const double next=std::min(tx,ty);
    if (next>exit || !std::isfinite(next)) { return true; }
    if (std::abs(tx-ty)<=1e-12*std::max(1.0,std::abs(next))) {
      if (blocked(x+sx,y) || blocked(x,y+sy)) { return false; }
      x+=sx; y+=sy; tx+=dx; ty+=dy;
    } else if (tx<ty) { x+=sx; tx+=dx; }
    else { y+=sy; ty+=dy; }
  }
  return false;
}
} // namespace

void validate_config(const Config &c) {
  const auto positive=[](double value){return std::isfinite(value)&&value>0;};
  if (!(std::isnan(c.ground_height_m)||positive(c.ground_height_m)) ||
      !positive(c.ground_height_tolerance_m) || !positive(c.auto_ground_height_min_m) ||
      !positive(c.auto_ground_height_max_m) || c.auto_ground_height_min_m>=c.auto_ground_height_max_m ||
      !positive(c.max_slope_deg) || c.max_slope_deg>=45 || !positive(c.fit_min_range_m) ||
      !positive(c.fit_max_range_m) || c.fit_min_range_m>=c.fit_max_range_m ||
      !positive(c.min_range_m) || !positive(c.max_range_m) || c.min_range_m>=c.max_range_m ||
      c.fit_min_range_m<c.min_range_m || c.fit_max_range_m>c.max_range_m ||
      !positive(c.ground_distance_m) || !positive(c.obstacle_min_height_m) ||
      c.obstacle_min_height_m<=c.ground_distance_m || !positive(c.obstacle_max_height_m) ||
      c.obstacle_max_height_m<=c.obstacle_min_height_m || !positive(c.patch_size_m) ||
      !positive(c.local_support_radius_m) || c.local_support_radius_m<c.patch_size_m ||
      c.min_ground_points<3 || c.min_ground_patches<3 || !positive(c.min_ground_span_m)) {
    throw std::invalid_argument("invalid ground geometry configuration");
  }
  for (const double v : {c.self_x_min_m,c.self_x_max_m,c.self_y_min_m,c.self_y_max_m,
                         c.self_z_min_m,c.self_z_max_m}) {
    if (!std::isfinite(v)) { throw std::invalid_argument("nonfinite self mask"); }
  }
  if (c.self_x_min_m>=c.self_x_max_m || c.self_y_min_m>=c.self_y_max_m ||
      c.self_z_min_m>=c.self_z_max_m) { throw std::invalid_argument("inverted self mask"); }
  // Grid performs its own bounds/resolution/allocation validation.
  const bev_core::Grid checked(c.grid);
  (void)checked;
}

Result classify_and_grid(const std::vector<Point> &points,const Config &config) {
  validate_config(config);
  Result result(config.grid);
  result.labels.resize(points.size(),Label::Unknown);
  result.heights_above_ground.resize(points.size(),std::numeric_limits<double>::quiet_NaN());
  Patches patches;
  for (std::size_t i=0;i<points.size();++i) {
    const auto &p=points[i];
    if (!finite(p) || (p.x==0 && p.y==0 && p.z==0)) { result.labels[i]=Label::Invalid; continue; }
    if (excluded_by_self_mask(p,config)) { result.labels[i]=Label::Masked; continue; }
    const double distance=std::hypot(std::hypot(p.x,p.y),p.z);
    if (distance<config.min_range_m || distance>config.max_range_m) { continue; }
    patches[patch_key(p,config.patch_size_m)].indices.push_back(i);
  }
  result.ground=estimate_floor(points,patches,config);
  if (!result.ground.valid) { return result; }
  // Patch offsets are bounded by half the ground tolerance: curbs cannot become a new floor.
  std::map<Key,LocalFloor> local;
  for (const auto &entry : patches) {
    std::vector<double> heights;
    for (const auto index : entry.second.indices) {
      const double d=residual(points[index],result.ground);
      if (std::abs(d)<=config.ground_distance_m) { heights.push_back(d); }
    }
    if (heights.size()<3) { continue; }
    const auto middle=heights.begin()+static_cast<std::ptrdiff_t>(heights.size()/2);
    std::nth_element(heights.begin(),middle,heights.end());
    const double limit=config.ground_distance_m*0.5;
    local.emplace(entry.first,LocalFloor{std::clamp(*middle,-limit,limit),heights.size()});
  }
  const int neighbor=static_cast<int>(std::ceil(config.local_support_radius_m/config.patch_size_m));
  for (const auto &entry : patches) {
    double local_offset=0, nearest=std::numeric_limits<double>::infinity();
    bool supported=false;
    for (int dx=-neighbor;dx<=neighbor;++dx) {
      for (int dy=-neighbor;dy<=neighbor;++dy) {
        const auto found=local.find({entry.first.first+dx,entry.first.second+dy});
        if (found==local.end()) { continue; }
        const double distance=std::hypot(static_cast<double>(dx),static_cast<double>(dy))*config.patch_size_m;
        if (distance<=config.local_support_radius_m && distance<nearest) {
          nearest=distance; local_offset=found->second.residual; supported=true;
        }
      }
    }
    if (!supported) { continue; }
    for (const auto index : entry.second.indices) {
      const auto &p=points[index];
      const double height=residual(p,result.ground)-local_offset;
      result.heights_above_ground[index]=height;
      if (std::abs(height)<=config.ground_distance_m) { result.labels[index]=Label::Ground; }
      else if (height>=config.obstacle_min_height_m && height<=config.obstacle_max_height_m) {
        result.labels[index]=Label::Obstacle;
        const auto cell=result.grid.index_for_point(p.x,p.y);
        if (cell) {
          auto &value=result.grid.cell(*cell);
          value.occupancy=bev_core::Occupancy::Occupied;
          value.ground_height_m=result.ground.a*p.x+result.ground.b*p.y+result.ground.c+local_offset;
          value.obstacle_height_m=std::isfinite(value.obstacle_height_m)?std::max(value.obstacle_height_m,height):height;
          value.confidence=1.0F;
        }
      }
    }
  }
  // Occupancy is complete before admitting any free evidence. Do not clear rays
  // through unseen spaces, nor infer traversability from elevated wall returns.
  std::vector<std::size_t> ground_endpoints(result.grid.cell_count(),points.size());
  for (std::size_t i=0;i<points.size();++i) {
    if (result.labels[i]!=Label::Ground) { continue; }
    const auto cell=result.grid.index_for_point(points[i].x,points[i].y);
    if (cell && ground_endpoints[*cell]==points.size()) { ground_endpoints[*cell]=i; }
  }
  for (std::size_t cell=0;cell<ground_endpoints.size();++cell) {
    const auto i=ground_endpoints[cell];
    if (i==points.size() || result.grid.cell(cell).occupancy==bev_core::Occupancy::Occupied ||
        !clear_ground_line(points[i],result.grid)) { continue; }
    auto &value=result.grid.cell(cell);
    value.occupancy=bev_core::Occupancy::Free;
    value.ground_height_m=points[i].z;
    value.confidence=1.0F;
  }
  return result;
}
} // namespace bev_geometry
