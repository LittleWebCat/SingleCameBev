#include "bev_path_tracking/pure_pursuit.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace bev_path_tracking;
void require(bool condition,const char *reason) { if(!condition) throw std::runtime_error(reason); }
std::vector<Pose> straight(double length=80) {
  std::vector<Pose> points;
  for(int i=0;i<=static_cast<int>(length);++i) points.push_back({double(i),0,0,0,1+.1*i});
  return points;
}
int main() {
  try {
    Parameters p; p.calibrated=true;
    Tracker tracker(straight(),p);
    tracker.update({1,1,0,0,100});
    auto left=tracker.update({1.1,1,0,0,100.1});
    require(left.valid&&left.targets_valid&&left.lateral_error>.9&&left.steering<0,
            "left-of-route must steer right");
    tracker.reset();
    tracker.update({1,-1,0,0,100});
    auto right=tracker.update({1.1,-1,0,0,100.1});
    require(right.targets_valid&&right.steering>0,"right-of-route must steer left");
    tracker.reset();
    tracker.update({1,0,0,0,100});
    auto center=tracker.update({1.1,0,0,0,100.1});
    require(center.targets_valid&&std::abs(center.steering)<1e-9,"straight route must not steer");
    require(!tracker.update({1.2,8,0,0,100.2}).targets_valid,"off-route must clear targets");
    tracker.reset();
    require(!tracker.update({1,0,0,3.141592653589793,100}).valid,"opposite heading must reject");
    Parameters unknown=p; unknown.calibrated=false;
    Tracker monitor(straight(),unknown);
    monitor.update({1,0,0,0,100});
    auto preview=monitor.update({1.1,0,0,0,100.1});
    require(preview.valid&&!preview.targets_valid,"unknown axle transform must withhold steering");
    Parameters lever=p; lever.lidar_forward=1.5; lever.lidar_left=.2; lever.sensor_yaw=.5;
    auto rear=vehicle_pose({1.5,.2,0,.5,100},lever);
    require(std::hypot(rear.x,rear.y)<1e-9&&std::abs(rear.yaw)<1e-9,"rear axle/yaw transform");
    Tracker endpoint(straight(20),p);
    endpoint.update({19.9,0,0,0,100});
    auto end=endpoint.update({20,0,0,0,100.1});
    require(end.finished&&end.targets_valid&&end.target_speed==0,"endpoint must stop");
    std::vector<Pose> circle;
    for(int i=0;i<=100;++i) {
      double angle=i*.01;
      circle.push_back({10*std::sin(angle),10*(1-std::cos(angle)),0,angle,1+.1*i});
    }
    Tracker curve(circle,p);
    curve.update({0,0,0,0,100});
    auto turn=curve.update({.1,0,0,.01,100.1});
    require(turn.targets_valid&&turn.steering>0,"left curve must steer left");
    // Observed curvature must come from motion, including lever-arm correction
    // and heading wrap, rather than from the Pure Pursuit target.
    for (double sign : {-1.,1.}) {
      Parameters measured=p; measured.lidar_forward=1.25; measured.lidar_left=.01;
      measured.sensor_yaw=.3;
      auto sensor_pose=[&](int index,double stamp) {
        double angle=sign*.02*index,yaw=3.10+angle;
        // Rotate the circle's initial tangent by 3.10 radians.
        double x=20*std::sin(sign*angle),y=sign*20*(1-std::cos(angle));
        Pose rear{x*std::cos(3.10)-y*std::sin(3.10),
                  x*std::sin(3.10)+y*std::cos(3.10),0,wrap(yaw),stamp};
        return Pose{rear.x+std::cos(yaw)*measured.lidar_forward-std::sin(yaw)*measured.lidar_left,
                    rear.y+std::sin(yaw)*measured.lidar_forward+std::cos(yaw)*measured.lidar_left,
                    0,wrap(yaw+measured.sensor_yaw),stamp};
      };
      std::vector<Pose> reference;
      for(int i=0;i<80;++i) reference.push_back(sensor_pose(i,1+.1*i));
      Tracker observed(reference,measured);
      for(int i=0;i<12;++i) {
        auto estimate=observed.update(sensor_pose(i,100+.1*i));
        if(i<4) require(!estimate.motion_valid,"motion estimate requires a warm window");
        else {
          require(estimate.motion_valid&&std::abs(estimate.vehicle_curvature-sign*.05)<1e-5,
                  "observed curvature sign/scale across yaw wrap and rear axle correction");
          require(std::abs(estimate.yaw_rate-sign*.2)<1e-6,"observed yaw rate");
        }
      }
      auto stopped=sensor_pose(11,101.2);
      require(!observed.update(stopped).motion_valid,"stop must clear observed curvature");
      observed.reset();
      require(!observed.update(sensor_pose(12,102)).motion_valid,"reset clears motion history");
      require(!observed.update(sensor_pose(13,103)).motion_valid,"pose gap restarts motion warmup");
    }
    Tracker stationary(straight(),p);
    for(int i=0;i<10;++i) require(!stationary.update({1,0,0,.001*i,100+.1*i}).motion_valid,
                                 "stationary yaw noise must not create curvature");
    Tracker straight_motion(straight(),p);
    Result straight_estimate;
    for(int i=0;i<10;++i) straight_estimate=straight_motion.update({1+.2*i,0,0,0,100+.1*i});
    require(straight_estimate.motion_valid&&std::abs(straight_estimate.vehicle_curvature)<1e-9,
            "straight motion has zero observed curvature");
    auto gap=straight(20);
    for(std::size_t i=10;i<gap.size();++i) gap[i].stamp+=2;
    Tracker split(gap,p);
    split.update({8,0,0,0,100});
    auto before_gap=split.update({8.1,0,0,0,100.1});
    require(before_gap.target.x<=9.00001&&before_gap.remaining<1,"must not bridge reference gap");
    std::vector<Pose> crossing={{0,0,0,0,1},{10,0,0,0,1.1},{10,10,0,1.57,1.2},
      {0,10,0,3.14,1.3},{0,0,0,-1.57,1.4},{10,0,0,0,1.5}};
    Tracker loop(crossing,p);
    loop.update({1,0,0,0,100});
    auto stay=loop.update({1.1,0,0,0,100.1});
    require(stay.valid&&stay.progress<5,"progress must not jump to later crossing");
    Tracker simulation(straight(),p);
    Pose vehicle{0,1,0,0,100};
    for(int i=0;i<150;++i) {
      auto target=simulation.update(vehicle);
      double steering=target.targets_valid?target.steering:0;
      vehicle.x+=.1*std::cos(vehicle.yaw);
      vehicle.y+=.1*std::sin(vehicle.yaw);
      vehicle.yaw=wrap(vehicle.yaw+.1*std::tan(steering)/p.wheelbase);
      vehicle.stamp+=.1;
    }
    require(std::abs(vehicle.y)<.15,"closed-loop bicycle must converge from lateral offset");
    bool failed=false;
    auto bad=straight(); bad[1].stamp=bad[0].stamp;
    try { Tracker invalid(bad,p); } catch(const std::exception&) { failed=true; }
    require(failed,"duplicate reference timestamps must reject");
    std::cout<<"Pure Pursuit: geometry, calibration gating, gap/crossing handling and bicycle convergence passed.\n";
    return 0;
  } catch(const std::exception &error) { std::cerr<<error.what()<<"\n"; return 1; }
}
