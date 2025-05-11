let validMovesVisible = true;

function updateScoresDisplay(blackScore, whiteScore) {
  document.getElementById('black-score-display').textContent = `Black: ${blackScore}`;
  document.getElementById('white-score-display').textContent = `White: ${whiteScore}`;
}

function updateGameBoardAndScores() {
  return fetch('/get_game_state')
    .then(response => response.json())
    .then(data => {
      if (data.message && data.message === 'Game instance not found') {
        console.log("Game instance not found, scores might be default.");
        updateScoresDisplay(2, 2);
        return null;
      }
      const gameState = data.game_state;
      updateScoresDisplay(data.black_score, data.white_score);

      for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
          const cell = gameState[row][col];
          const cellElement = document.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
          cellElement.innerHTML = '';
          if (cell === 'BLACK') {
            cellElement.innerHTML = '<div class="black-disc"></div>';
          } else if (cell === 'WHITE') {
            cellElement.innerHTML = '<div class="white-disc"></div>';
          } else if (cell === 'VALID') {
            if (validMovesVisible && data.active_player_type === 'user' && !data.is_finished) {
              cellElement.innerHTML = '<div class="grey-disc"></div>';
            }
          }
        }
      }
      return data;
    })
    .catch(error => {
      console.error('Error fetching game state:', error);
      return null;
    });
}

function displayMessage(message) {
  const messageBox = document.getElementById('message-box');
  messageBox.textContent = message;
  messageBox.style.visibility = 'visible';
}

function fetchAndDisplayGameOutcome() {
  fetch('/get_game_outcome')
    .then(response => response.json())
    .then(data => {
      displayMessage(data.outcome_message);
      if (data.black_score !== undefined && data.white_score !== undefined) {
        updateScoresDisplay(data.black_score, data.white_score);
      }
      validMovesVisible = false;
      updateGameBoardAndScores();
    })
    .catch(error => console.error('Error fetching game outcome:', error));
}

function handleUserMove(event) {
  const tdElement = event.target.closest('.cell');
  if (!tdElement) {
    console.error('Invalid click, no parent <td> element found.');
    return;
  }

  const row = parseInt(tdElement.dataset.row);
  const col = parseInt(tdElement.dataset.col);

  document.getElementById('message-box').style.visibility = 'hidden';
  validMovesVisible = false;

  fetch('/user_move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ row, col }),
  })
  .then(response => response.json())
  .then(data => {
    console.log('Response from backend (user_move):', data);
    if (data.error) {
        displayMessage(data.message);
        validMovesVisible = true; 
        updateGameBoardAndScores();
        return;
    }
    updateGameBoardAndScores().then(gameStateData => {
        if (data.game_over) {
            fetchAndDisplayGameOutcome();
        } else if (data.next_player_is_ai) {
            handleAgentMove();
        } else {
             validMovesVisible = true; 
             updateGameBoardAndScores();
        }
    });
  })
  .catch(userError => {
    console.error('User Error:', userError);
    validMovesVisible = true;
    updateGameBoardAndScores();
  });
}

function handleAgentMove() {
  validMovesVisible = false;
  updateGameBoardAndScores();
  displayMessage("OthelloAI is analysing...");

  setTimeout(() => {
    fetch('/agent_move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    })
    .then(response => response.json())
    .then(data => {
      console.log('Response from backend (agent_move):', data);
       if (data.error) {
          displayMessage(data.message);
          updateGameBoardAndScores();
          return;
      }

      updateGameBoardAndScores().then(gameStateData => {
        if (data.game_over) {
          fetchAndDisplayGameOutcome();
        } else {
          if (data.current_player_has_moves) {
            if (data.current_player_is_ai) {
              displayMessage("OthelloAI is analysing...");
              setTimeout(handleAgentMove, 1000); 
            } else { 
              validMovesVisible = true;
              document.getElementById('message-box').style.visibility = 'hidden';
              updateGameBoardAndScores();
            }
          } else { 
            if (data.current_player_is_ai){
                 displayMessage("AI has no moves, passing turn. Other AI is thinking...");
            } else {
                 displayMessage("AI has no moves, passing turn. Your turn.");
            }
            setTimeout(handleAgentMove, 1000);
          }
        }
      });
    })
    .catch(agentError => {
      console.error('Agent Error:', agentError);
      displayMessage("Error during AI move. Please try again or reset.");
      updateGameBoardAndScores();
    });
  }, 1000);
}

function resetGame() {
  fetch('/reset_game', {
    method: 'POST',
  })
  .then(() => {
    location.reload();
  })
  .catch(error => {
    console.error('Error resetting the game:', error);
  });
}

document.addEventListener('DOMContentLoaded', function () {
  const gameTable = document.querySelector('.game-board');
  if (gameTable) {
    gameTable.addEventListener('click', function (event) {
      if (event.target.closest('.grey-disc')) {
        handleUserMove(event);
      }
    });
  }

  const resetGameButton = document.getElementById('reset-game-button');
  if (resetGameButton) {
    resetGameButton.addEventListener('click', resetGame);
  }
  
  const gameStartedFlag = document.getElementById('game-started').dataset.gameStarted === 'True';
  const playerBlackType = document.getElementById('game-started').dataset.playerBlackType;

  if (gameStartedFlag) {
    updateGameBoardAndScores().then(gameStateData => {
        if (gameStateData && !gameStateData.is_finished) {
            if (gameStateData.active_player_type !== 'user') {
                 let pieceCount = 0;
                 document.querySelectorAll('.black-disc, .white-disc').forEach(() => pieceCount++);
                 if (pieceCount <= 4 || gameStateData.active_player_color === 'BLACK') {
                    console.log("Game started. Current active player is AI. AI is making a move.");
                    handleAgentMove();
                 } else {
                    validMovesVisible = false; 
                    updateGameBoardAndScores();
                 }
            } else {
                validMovesVisible = true;
                updateGameBoardAndScores();
            }
        } else if (gameStateData && gameStateData.is_finished) {
            fetchAndDisplayGameOutcome();
        }
    });
  } else {
    updateGameBoardAndScores(); 
  }
});