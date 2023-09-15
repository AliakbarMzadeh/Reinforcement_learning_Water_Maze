% Water Maze Task
% Implementation in MATLAB

% Parameters
mapSize = 15;  % Size of the map
numTrials = 100;  % Number of trials
eta = 0.1;  % Learning rate
gamma = 0.9;  % Discount factor

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
        
        % Choose the action with the highest value (or randomly among equal values)
        nextAction = chooseAction(currentState, actions, V);
        
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
    subplot(1, 3, 1:2);
    plot(path(:, 2), path(:, 1), 'Color', 'k');
    hold on;
    
    % Plot the start point as a yellow point
    plot(path(1, 2), path(1, 1), 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k');
    
    % Plot the Reward and Punishment locations
    subplot(1, 3, 3);
    imagesc(V);
    colorbar;
    hold on;
    plot(target(2), target(1), 'sk', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    plot(cat(2), cat(1), 'sr', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    hold off;
    
    % Display the state values
    title(sprintf('State Values (Trial %d)', trial));
    xlabel('Column');
    ylabel('Row');
    
    % Pause for a brief moment to visualize each trial
    pause(0.5);
end

% Calculate and plot the gradients of learned values
gradients = diff(V, 1, 1);
gradients = [zeros(1, size(gradients, 2)); gradients];
subplot(1, 3, 3);
hold on;
quiver(1:mapSize, 1:mapSize, -gradients, zeros(size(gradients)), 'Color', 'w', 'LineWidth', 1);
hold off;

% Plot the final State Values after training
subplot(1, 3, 3);
title('Final State Values');
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
