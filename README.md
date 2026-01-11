# 2D Circle Region Identifier

This project implements an algorithm to count the number of regions formed by intersecting circles on a 2D plane and determines whether each region is inside or outside each circle. The results are visualized using `matplotlib`, with plots saved in the `.nosync` directory.

## Setup

1. **Clone the Repository**:
git clone https://github.com/DDDOH/2D-Circle-Region-Identifier.git
cd 2D-Circle-Region-Identifier

2. **Create a Virtual Environment**:
`python -m venv venv`
Activate it:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

3. **Install Dependencies**:
`pip install -r requirements.txt`

4. **Run the Script**:
`python main.py`


## Usage

- The script generates random circles (default: 20) and computes intersecting regions, saving plots to `.nosync/region_X.png`.
- Modify parameters in `main.py`:
- `n_circle`: Number of circles (e.g., set to 3 for testing).
- `PLOT_ARC = True`: Visualize arcs for debugging.
- `PLOT_POINT = True`: Show intersection points.
- `PLOT_REGION = True`: Plot detected regions (default).
- Example: Set `n_circle = 3` and run `python main.py` to visualize regions for three circles.

## Algorithm Overview

The algorithm:
1. Generates random circles with centers `(c_x, c_y)` and radii `r`.
2. Computes intersection points between all circle pairs using `intersect_two_circle`.
3. Identifies arcs (circle segments between intersections) and assigns them to regions.
4. Traverses arcs to delineate regions, determining their inside/outside relationship with each circle.
5. Visualizes results with `matplotlib`, showing circles and filled regions.

## Contributing

1. Fork the repository and clone it locally.
2. Create a branch for your changes: `git checkout -b your-branch-name`.
3. Make changes, test with `python main.py`, and commit: `git commit -m "Your change description"`.
4. Push to your fork: `git push origin your-branch-name`.
5. Open a pull request to the main repository.

Suggestions for contributions:
- Add unit tests for `utils.py` functions.
- Improve error handling for edge cases (e.g., non-intersecting circles).
- Enhance visualizations with labels or legends.

## License

No license specified. Contact the repository owner for usage permissions.
