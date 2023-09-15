% Water Maze Task
% Implementation in MATLAB

% Parameters
mapSize = 15;  % Size of the map
numTrials = 100;  % Number of trials
eta = 0.2;  % Learning rate
gamma = 0.9;  % Discount factor
epsilon = 0.7;  % Exploration rate (greedy epsilon)

% Initialize the map
map = zeros(mapSize, mapSize);  % Create a map of zeros
target = [3, 12];  % Target location (black)
cat = [8, 8];  % Cat location (red)

% Initialize the state values
V = zeros(mapSize, mapSize);  % Initialize all state values to zero
V(target(1), target(2)) = 10;  % Set the target state value to reward
V(cat(1), cat(2)) = -10;  % Set the cat state value to punishment

% Run the trials
figure;

for trial = 1:numTrials
    % Randomly select the starting position
    start = randi([2, mapSize-1], 1, 2);  % Exclude the edges
    
    % Simulate agent's movement until reaching the target or cat
    currentState = start;
    path = currentState;
    
    while ~isequal(currentState, target) && ~isequal(currentState, cat)
        % Get the available actions
        actions = getAvailableActions(currentState, mapSize);
        
        % Choose the action with the highest value with probability epsilon
        if rand <= epsilon
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
    end
    
    % Plot the path for this trial
    if ismember(trial, [1, 12, 20, 25])
        figure;
        plot(path(:, 2), path(:, 1), 'Color', 'k');
        hold on;
        plot(start(2), start(1), 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k');
        plot(target(2), target(1), 'sk', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
        plot(cat(2), cat(1), 'sr', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        hold off;
        title(sprintf('Path (Trial %d)', trial));
        xlabel('Column');
        ylabel('Row');
    end
end

% Plot the gradients of learned values
figure;
[X, Y] = meshgrid(1:mapSize);
quiver(X, Y, -gradient(V)', -gradient(V), 'Color', 'b');
hold on;
plot(target(2), target(1), 'sk', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
plot(cat(2), cat(1), 'sr', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
hold off;
title('Gradients of Learned Values');
xlabel('Column');
ylabel('Row');

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
