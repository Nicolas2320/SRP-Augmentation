param(
    [ValidateSet('scratch', 'pretrained')][string]$Initialization = 'scratch',
    [ValidateSet(20, 50, 100, 450)][int]$K = 20,
    [ValidateSet('raw', 'none', 'mixup', 'cutmix', 'augmix', 'simmixup', 'simcutmix')]
    [string]$Method = 'simcutmix',
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$pythonPath = Join-Path (Get-Location) '.venv/Scripts/python.exe'
$epochCount = 100
$milestones = @('30', '55', '75')
$learningRate = '0.01'
$lrGamma = '0.1'
if ($K -eq 450) {
    $epochCount = 50
    $milestones = @('15', '30', '40')
    $learningRate = '0.1'
    $lrGamma = '0.2'
}
$initFlag = '--pretrained'
if ($Initialization -eq 'scratch') { $initFlag = '--no-pretrained' }
$trainArgs = @(
    '-u', 'src/train.py', '--dataset', 'cifar100', '--model', 'resnet50',
    $initFlag, '--k', "$K", '--subset-seed', '0', '--train-seed', '0',
    '--augmentation', $Method, '--epochs', "$epochCount", '--batch-size', '32',
    '--optimizer', 'sgd', '--lr', $learningRate, '--momentum', '0.9', '--nesterov',
    '--weight-decay', '0.0005', '--lr-milestones'
) + $milestones + @(
    '--lr-gamma', $lrGamma, '--mixup-alpha', '1', '--cutmix-alpha', '1',
    '--cutmix-prob', '0.5', '--mix-prob', '1', '--mix-warmup-epochs', '0',
    '--num-workers', '2',
    '--output-root', "results/comparison_v1/$Initialization"
)
if ($Method -in @('simmixup', 'simcutmix')) {
    $neighbors = "results/experiments/shared/neighbors/cifar100/k${K}_seed0/neighbors_class_agnostic_K40.pt"
    if (-not (Test-Path -LiteralPath $neighbors)) { throw "Missing neighbors: $neighbors" }
    $trainArgs += @('--neighbor-path', $neighbors, '--guided-mode', 'class_agnostic',
        '--neighbor-k', '20', '--neighbor-rank-start', '21', '--pair-sampling', 'uniform')
}
Write-Output (('& "{0}" ' -f $pythonPath) + (($trainArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '))
if ($Execute) {
    & $pythonPath @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
}
