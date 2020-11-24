function [plane, min_outliers] = ransac_find_plane(pts, threshold)
%       pts: 3xN, N 3D points
% threshold: Scalar, threshold for points to be inliers
%     plane: 4x1, plane on the form ax+by+cz+d=0
% 

if threshold == 0
    warning('Threshold = 0 may give false outliers due to machine precision errors')
end

% Init
N = size(pts,2);
epsilon = 0.4; % epsilon0

mismatch_prob = 0.3; % eta

kmax = log(mismatch_prob)/log(1-epsilon^3);
min_outliers = N;
k=1;

while k < kmax
%for i = 1:kmax
    
    % %%%%% Select subset of points and calculate preliminary plane %%%%
    
    subset = randperm(N,3); % randomize 3 points
    pts_prim = pts(:,subset);
    
    plane_prel = compute_plane(pts_prim);

    % %%%%% Measure performance of prel plane %%%%%%
    residual_lengths = residual_lengths_points_to_plane(pts,plane_prel);
    
    outliers = sum(residual_lengths > threshold);
    inliers = N-outliers;
    
    % %%%%% Log keeping %%%%%
    if (outliers > 0)
        if outliers < min_outliers
            % if best sub-perfect case found, save loss and iterate
            min_outliers = outliers;

            plane = plane_prel;
            
            epsilon = inliers/N;
            kmax = log(mismatch_prob)/log(1-epsilon^3);
            
        end
    else
        % If best case found (outliers = 0), return immidiately
        warning('# of outliers !> 0. (this case is not yet tested)')
        plane = plane_prel;
        min_outliers = outliers;
        disp(['Total # of iterations was ' num2str(k) ' with 0% outliers.' ])
        return
    end
    
    k=k+1;

end % main whileloop

disp(['Total # of iterations was ' num2str(k) ' and optimal percentage of outliers was ' num2str((min_outliers/N)*100) '%.'])




end

