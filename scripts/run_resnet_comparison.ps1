param(
    [ValidateSet('scratch', 'pretrained')][string]$Initialization = 'scratch',
    [ValidateSet(20, 50, 100, 200, 300, 450)][int]$K = 20,
    [ValidateSet(0, 1, 2)][int]$SubsetSeed = 0,
    [ValidateRange(0, [int]::MaxValue)][int]$TrainSeed = 0,
    [ValidateSet('raw', 'none', 'mixup', 'cutmix', 'augmix', 'simmixup', 'simcutmix')]
    [string]$Method = 'simcutmix',
    [switch]$SkipTest,
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$splitPath = "data/splits/cifar100/k${K}_seed${SubsetSeed}.json"
if ($K -eq 450 -and $SubsetSeed -ne 0) {
    throw 'k=450 is the complete post-validation pool and only has subset seed 0.'
}
if (-not (Test-Path -LiteralPath $splitPath)) { throw "Missing split: $splitPath" }
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
$outputRoot = "results/third_presentation/final/$Initialization"
if ($SkipTest) { $outputRoot = "results/third_presentation/screening/$Initialization" }
$trainArgs = @(
    '-u', 'src/train.py', '--dataset', 'cifar100', '--model', 'resnet50',
    $initFlag, '--k', "$K", '--subset-seed', "$SubsetSeed", '--train-seed', "$TrainSeed",
    '--augmentation', $Method, '--epochs', "$epochCount", '--batch-size', '32',
    '--optimizer', 'sgd', '--lr', $learningRate, '--momentum', '0.9', '--nesterov',
    '--weight-decay', '0.0005', '--lr-milestones'
) + $milestones + @(
    '--lr-gamma', $lrGamma, '--mixup-alpha', '1', '--cutmix-alpha', '1',
    '--cutmix-prob', '0.5', '--mix-prob', '1', '--mix-warmup-epochs', '0',
    '--num-workers', '2',
    '--output-root', $outputRoot
)
if ($SkipTest) { $trainArgs += '--skip-test' }
if ($Method -in @('simmixup', 'simcutmix')) {
    $neighbors = "results/experiments/shared/neighbors/cifar100/k${K}_seed${SubsetSeed}/neighbors_class_agnostic_K40.pt"
    if (-not (Test-Path -LiteralPath $neighbors)) { throw "Missing neighbors: $neighbors" }
    $neighborK = '20'
    $neighborRankStart = '21'
    if ($Method -eq 'simmixup') {
        $neighborK = '10'
        $neighborRankStart = '1'
    }
    $trainArgs += @('--neighbor-path', $neighbors, '--guided-mode', 'class_agnostic',
        '--neighbor-k', $neighborK, '--neighbor-rank-start', $neighborRankStart,
        '--pair-sampling', 'uniform')
}
Write-Output (('& "{0}" ' -f $pythonPath) + (($trainArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '))
if ($Execute) {
    & $pythonPath @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
}
