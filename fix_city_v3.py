"""Regen city figures: bigger panels, no title overlap."""
import sys, os, pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
_p = os.path.dirname(__file__)
_out = os.path.join(_p, 'outputs')
sys.path.append(os.path.join(_p, 'src'))
from src.utils import compute_distance_matrix, tour_cost
from city_delivery_scenario import generate_city_scenario

pts = generate_city_scenario(500, seed=42)
dist_mat = compute_distance_matrix(pts)
with open(os.path.join(_p, 'outputs', '_city_tours.pkl'), 'rb') as f:
    tours = pickle.load(f)

hoods = [
    ('North',(0.35,0.65,0.30,0.25),'lightcoral'),
    ('East',(0.70,0.45,0.15,0.20),'lightblue'),
    ('South',(0.30,0.05,0.25,0.20),'lightgreen'),
    ('West',(0.05,0.40,0.20,0.30),'lightyellow'),
    ('Biz',(0.40,0.45,0.10,0.15),'plum'),
]
def draw(ax, tour, title, cost):
    for _,rc,cl in hoods:
        ax.add_patch(plt.Rectangle((rc[0],rc[1]),rc[2],rc[3],fill=True,facecolor=cl,alpha=0.1,edgecolor='gray',lw=0.3))
    ax.scatter(pts[1:,0],pts[1:,1],c='steelblue',s=2,alpha=0.5,zorder=2)
    ax.scatter(pts[0,0],pts[0,1],c='red',s=120,marker='*',edgecolors='darkred',lw=1.5,zorder=5)
    for i in range(len(tour)-1):
        ax.plot([pts[tour[i],0],pts[tour[i+1],0]],[pts[tour[i],1],pts[tour[i+1],1]],'-',c='darkblue',lw=0.2,alpha=0.4)
    ax.set_title(title+'\nCost: %.2f'%cost, fontsize=14, fontweight='bold', pad=8)
    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02); ax.set_aspect('equal'); ax.axis('off')

# Page 1: 2x2
fig, axes = plt.subplots(2, 2, figsize=(18, 17))
plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.03, wspace=0.08, hspace=0.12)
for idx,(k,name) in enumerate([('NN','Nearest Neighbor'),('Christofides','Christofides'),('C+2opt','C+2opt'),('DIFUSCO','DIFUSCO+2opt')]):
    draw(axes[idx//2,idx%2], tours[k], name, tour_cost(dist_mat, tours[k]))
fig.suptitle('City-Wide Package Delivery -- 500 Locations', fontsize=20, fontweight='bold', y=0.97)
fig.savefig(os.path.join(_out, 'city_delivery_p1.png'), dpi=200)
plt.close(fig)

# Page 2: 1x2
fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.05, wspace=0.08)
for idx,(k,name) in enumerate([('DualOpt','DualOpt'),('LKH3','LKH3')]):
    draw(axes[idx], tours[k], name, tour_cost(dist_mat, tours[k]))
fig.suptitle('City-Wide Package Delivery -- 500 Locations (continued)', fontsize=20, fontweight='bold', y=0.96)
fig.savefig(os.path.join(_out, 'city_delivery_p2.png'), dpi=200)
plt.close(fig)
print('Done')
