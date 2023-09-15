% Water Maze Task
% Implementation in MATLAB

% Parameters
mapSize = 15;  % Size of the map
numTrials = 50;  % Number of trials
eta = 0.9;  % Learning rate
gamma = 0.9;  % Discount factor
epsilon = 0.97;  % Exploration rate (greedy epsilon)

% Initialize the map
map = zeros(mapSize, mapSize);  % Create a map of zeros
target = [14, 14];  % Target location (black)
cat = [2, 2];  % Cat location (red)

% Initialize the state values
V = zeros(mapSize, mapSize);  % Initialize all state values to zero
V(target(1), target(2)) = 10;  % Set the target state value to reward
V(cat(1), cat(2)) = -10;  % Set the cat state value to punishment



% Create figure for animation
figure;



 
    
    
    
    
    
    
for trial = 1:numTrials
    % Randomly select the starting position
    start = randi([2, mapSize-1], 1, 2);  % Exclude the edges
    
    % Determine the exploration rate for the current trial
    if trial <= numTrials / 2
        currentEpsilon = epsilon;  % Epsilon-greedy exploration
    else
        currentEpsilon = 0;  % No exploration, choose the action with the highest value
    end
    
    % Simulate agent's movement until reaching the target or cat
    currentState = start;
    path = currentState;
    
    while ~isequal(currentState, target) && ~isequal(currentState, cat)
        % Get the available actions
        actions = getAvailableActions(currentState, mapSize);
        
        % Choose the action based on the exploration rate
        if rand <= currentEpsilon
            nextAction = chooseAction(currentState, actions, V);  % Choose action based on highest value
        else
            nextAction = actions(randi(size(actions, 1)), :);  % Select a random non-greedy action
        end
        
        % Take the selected action and transition to the next state
        nextState = transitionState(currentState, nextAction);
        
        % Determine the immediate reward based on the next state
        immediateReward = getImmediateReward(nextState, target, cat);
        
        % Update the value of the current state
        delta = calculateDelta(immediateReward, currentState, nextState, gamma, V);
        V(currentState(1), currentState(2)) = V(currentState(1), currentState(2)) + eta * delta;
        
        % Move to the next state
        currentState = nextState;
        path = [path; currentState];
    
    
        
        % Plot the path and state values
        plot(path(:, 2), path(:, 1), 'Color', 'k');
        hold on;
        plot(path(1, 2), path(1, 1), 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k');
        plot(target(2), target(1), 'sk', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
        plot(cat(2), cat(1), 'sr', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        hold off;
        title(sprintf('Trial %d: Agent Path', trial));
        
        % Check if agent reached the cat state
        if isequal(currentState, cat)
            break;  % Terminate the current trial
        end

        % Pause briefly to visualize each transition
        pause(0.01);
    end

    % Reset the current state for the next trial
    currentState = start;
end

% Function to get available actions for a given state
function actions = getAvailableActions(state, mapSize)
    actions = [];
    if state(1) > 1
        actions = [actions; -1, 0];  % Up
    end
    if state(1) < mapSize
        actions = [actions; 1, 0];  % Down
    end
    if state(2) > 1
        actions = [actions; 0, -1];  % Left
    end
    if state(2) < mapSize
        actions = [actions; 0, 1];  % Right
    end
end

% Function to choose the action with the highest value (or randomly among equal values)
function action = chooseAction(state, actions, V)
    actionValues = zeros(size(actions, 1), 1);

    for i = 1:size(actions, 1)
        nextPos = state + actions(i, :);
        actionValues(i) = V(nextPos(1), nextPos(2));
    end

    [~, maxIdx] = max(actionValues);
    maxValuesIdx = find(actionValues == actionValues(maxIdx));
    action = actions(maxValuesIdx(randi(numel(maxValuesIdx))), :);
end

% Function to transition from the current state to the next state
function nextState = transitionState(currentState, action)
    nextState = currentState + action;
end

% Function to get the immediate reward based on the next state
function immediateReward = getImmediateReward(nextState, target, cat)
    if isequal(nextState, target)
        immediateReward = 10;  % Reward stage
    elseif isequal(nextState, cat)
        immediateReward = -10;  % Punishment stage
    else
        immediateReward = 0;
    end
end

% Function to calculate the delta value for updating the state value
function delta = calculateDelta(immediateReward, currentState, nextState, gamma, V)
    nextStateValue = V(nextState(1), nextState(2));
    delta = immediateReward + gamma * nextStateValue - V(currentState(1), currentState(2));
end
