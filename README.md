# Dots and Boxes Game

## Overview

This project is an implementation of the classic Dots and Boxes game, featuring a player versus AI gameplay. The game is built using Python with Flask for the backend and HTML/CSS/JavaScript for the frontend. It showcases an intelligent AI opponent using the Minimax algorithm with alpha-beta pruning.

## Features

- Interactive web-based interface
- Player vs AI gameplay
- Three AI difficulty levels: Easy, Medium, and Hard
- Minimax algorithm with alpha-beta pruning for AI decision making
- Responsive design with modern styling
- Sound effects for enhanced user experience
- Score tracking and game state management

## Technologies Used

- Backend: Python, Flask
- Frontend: HTML5, CSS3, JavaScript
- AI Algorithm: Minimax with alpha-beta pruning

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/dots-and-boxes.git
   cd dots-and-boxes
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install flask
   ```

4. Set up the static files:
   - Create a `static` folder in the project root if it doesn't exist.
   - Add sound effect files to the `static` folder:
     - `connect-sound.mp3`
     - `box-complete-sound.mp3`
     - `game-over-sound.mp3`

## Running the Game

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5001`

3. Enjoy playing Dots and Boxes against the AI!

## How to Play

1. The game starts with an empty grid of dots.
2. Players take turns connecting two adjacent dots with a line (horizontal or vertical).
3. When a player completes a box, they score a point and get another turn.
4. The game ends when all possible lines have been drawn.
5. The player with the most completed boxes wins.

## AI Difficulty Levels

- Easy: The AI makes random moves.
- Medium: The AI uses the Minimax algorithm with a depth of 2.
- Hard: The AI uses the Minimax algorithm with a depth of 3 (or more, adjustable in the code).

## Contributing

Contributions to improve the game are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is open-source and available under the MIT License.
